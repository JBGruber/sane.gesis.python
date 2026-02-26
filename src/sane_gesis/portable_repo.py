"""Build a Portable Package Repository from Python Project Files.

Scans Python scripts and Jupyter notebooks in a directory to detect package
dependencies, downloads all required packages and their dependencies, and
creates a compressed portable package repository.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

# Common import name -> package name mappings
# (import name on the left, pip package name on the right)
IMPORT_TO_PACKAGE = {
    "cv2": "opencv-python",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "git": "gitpython",
    "serial": "pyserial",
    "usb": "pyusb",
    "wx": "wxpython",
    "Crypto": "pycryptodome",
    "OpenSSL": "pyopenssl",
    "jwt": "pyjwt",
    "magic": "python-magic",
    "lxml": "lxml",
    "Bio": "biopython",
    "cv": "opencv-python",
    "fitz": "pymupdf",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "xlrd": "xlrd",
    "openpyxl": "openpyxl",
}

# Standard library modules (Python 3.10+)
# Using sys.stdlib_module_names when available (3.10+)
if hasattr(sys, "stdlib_module_names"):
    STDLIB_MODULES = sys.stdlib_module_names
else:
    # Fallback for older Python versions
    STDLIB_MODULES = {
        "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio",
        "asyncore", "atexit", "audioop", "base64", "bdb", "binascii",
        "binhex", "bisect", "builtins", "bz2", "calendar", "cgi", "cgitb",
        "chunk", "cmath", "cmd", "code", "codecs", "codeop", "collections",
        "colorsys", "compileall", "concurrent", "configparser", "contextlib",
        "contextvars", "copy", "copyreg", "cProfile", "crypt", "csv",
        "ctypes", "curses", "dataclasses", "datetime", "dbm", "decimal",
        "difflib", "dis", "distutils", "doctest", "email", "encodings",
        "enum", "errno", "faulthandler", "fcntl", "filecmp", "fileinput",
        "fnmatch", "fractions", "ftplib", "functools", "gc", "getopt",
        "getpass", "gettext", "glob", "graphlib", "grp", "gzip", "hashlib",
        "heapq", "hmac", "html", "http", "idlelib", "imaplib", "imghdr",
        "imp", "importlib", "inspect", "io", "ipaddress", "itertools",
        "json", "keyword", "lib2to3", "linecache", "locale", "logging",
        "lzma", "mailbox", "mailcap", "marshal", "math", "mimetypes",
        "mmap", "modulefinder", "multiprocessing", "netrc", "nis",
        "nntplib", "numbers", "operator", "optparse", "os", "ossaudiodev",
        "pathlib", "pdb", "pickle", "pickletools", "pipes", "pkgutil",
        "platform", "plistlib", "poplib", "posix", "posixpath", "pprint",
        "profile", "pstats", "pty", "pwd", "py_compile", "pyclbr",
        "pydoc", "queue", "quopri", "random", "re", "readline", "reprlib",
        "resource", "rlcompleter", "runpy", "sched", "secrets", "select",
        "selectors", "shelve", "shlex", "shutil", "signal", "site",
        "smtpd", "smtplib", "sndhdr", "socket", "socketserver", "spwd",
        "sqlite3", "ssl", "stat", "statistics", "string", "stringprep",
        "struct", "subprocess", "sunau", "symtable", "sys", "sysconfig",
        "syslog", "tabnanny", "tarfile", "telnetlib", "tempfile", "termios",
        "test", "textwrap", "threading", "time", "timeit", "tkinter",
        "token", "tokenize", "trace", "traceback", "tracemalloc", "tty",
        "turtle", "turtledemo", "types", "typing", "unicodedata", "unittest",
        "urllib", "uu", "uuid", "venv", "warnings", "wave", "weakref",
        "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib", "xml",
        "xmlrpc", "zipapp", "zipfile", "zipimport", "zlib", "_thread",
    }


def extract_imports_from_source(source: str) -> set[str]:
    """Parse Python source code and extract top-level import names."""
    imports = set()
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:  # absolute imports only
                    imports.add(node.module.split(".")[0])
    except SyntaxError:
        pass
    return imports


def extract_imports_from_file(py_file: Path) -> set[str]:
    """Parse a Python file and extract import statements."""
    try:
        source = py_file.read_text(encoding="utf-8")
        return extract_imports_from_source(source)
    except (OSError, UnicodeDecodeError):
        return set()


def extract_imports_from_notebook(nb_file: Path) -> set[str]:
    """Extract imports from a Jupyter notebook."""
    imports = set()
    try:
        content = json.loads(nb_file.read_text(encoding="utf-8"))
        for cell in content.get("cells", []):
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                imports.update(extract_imports_from_source(source))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        pass
    return imports


def _get_wheel_requires_python(whl_path: Path) -> str | None:
    """Extract the Requires-Python field from a wheel's METADATA."""
    try:
        with zipfile.ZipFile(whl_path) as zf:
            metadata_names = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
            if not metadata_names:
                return None
            metadata = zf.read(metadata_names[0]).decode("utf-8", errors="replace")
            for line in metadata.splitlines():
                if line.lower().startswith("requires-python:"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def _is_python_version_compatible(requires_python: str, python_version: str) -> bool:
    """Return True if python_version satisfies the Requires-Python specifier."""
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
        return Version(python_version) in SpecifierSet(requires_python)
    except ImportError:
        pass
    # Fallback: manual tuple comparison
    def _ver(s: str) -> tuple[int, ...]:
        return tuple(int(x) for x in s.split(".")[:3])

    target = _ver(python_version)
    for spec in requires_python.split(","):
        spec = spec.strip()
        m = re.match(r"([><=!~]+)\s*(\d[\d.]*)", spec)
        if not m:
            continue
        op, ver_str = m.group(1), m.group(2)
        req = _ver(ver_str)
        n = max(len(target), len(req))
        t = target + (0,) * (n - len(target))
        r = req + (0,) * (n - len(req))
        if op == ">=" and not t >= r:
            return False
        elif op == ">" and not t > r:
            return False
        elif op == "<=" and not t <= r:
            return False
        elif op == "<" and not t < r:
            return False
        elif op == "==" and not t == r:
            return False
        elif op == "!=" and t == r:
            return False
        elif op == "~=" and not (t >= r and t[0] == r[0]):
            return False
    return True


def _find_latest_compatible_version(package_name: str, python_version: str) -> str | None:
    """Query PyPI to find the latest release of package compatible with python_version.

    Returns a version string (e.g. "0.10.4") or None if nothing suitable is found.
    """
    import urllib.request

    name = re.split(r"[><=!~\[]", package_name)[0].strip()
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

    releases = data.get("releases", {})

    try:
        from packaging.version import Version

        def _ver_key(v: str) -> "Version":
            try:
                return Version(v)
            except Exception:
                return Version("0")

        sorted_versions = sorted(releases.keys(), key=_ver_key, reverse=True)
    except ImportError:
        sorted_versions = sorted(releases.keys(), reverse=True)

    for ver_str in sorted_versions:
        files = releases[ver_str]
        if not files:
            continue
        req_python = next((f["requires_python"] for f in files if f.get("requires_python")), None)
        if req_python is None or _is_python_version_compatible(req_python, python_version):
            return ver_str

    return None


def _platform_to_sys_platform(platform: str) -> str:
    """Map a pip platform tag to the sys.platform value used in environment markers."""
    if platform.startswith("win"):
        return "win32"
    if "linux" in platform:
        return "linux"
    if "macosx" in platform or "darwin" in platform:
        return "darwin"
    return "linux"


def _eval_marker(marker_str: str, python_version: str, sys_platform: str) -> bool:
    """Evaluate a PEP 508 environment marker for a specific target python_version/platform.

    pip download --python-version X does NOT evaluate python_version markers against X;
    it uses the running interpreter's version. This function fills that gap.
    """
    ver_parts = python_version.split(".")
    py_ver = ".".join(ver_parts[:2])
    env = {
        "python_version": py_ver,
        "python_full_version": python_version if len(ver_parts) >= 3 else python_version + ".0",
        "implementation_name": "cpython",
        "implementation_version": python_version if len(ver_parts) >= 3 else python_version + ".0",
        "platform_python_implementation": "CPython",
        "sys_platform": sys_platform,
        "os_name": "nt" if sys_platform == "win32" else "posix",
        "platform_machine": "AMD64" if sys_platform == "win32" else "x86_64",
        "platform_system": (
            "Windows" if sys_platform == "win32"
            else ("Darwin" if sys_platform == "darwin" else "Linux")
        ),
        "extra": "",
    }
    try:
        from packaging.markers import Marker
        return Marker(marker_str).evaluate(env)
    except Exception:
        pass
    # Fallback: handle the common python_version comparison case
    m = re.search(r'python_version\s*([<>=!]+)\s*["\']?([\d.]+)["\']?', marker_str)
    if m:
        op, ver = m.group(1), m.group(2)
        return _is_python_version_compatible(f"{op}{ver}", python_version)
    return True  # conservative: assume required if marker can't be parsed


def _get_marker_required_deps(whl_path: Path, python_version: str, sys_platform: str) -> list[str]:
    """Return package names from Requires-Dist that are conditionally required.

    Only returns deps whose environment marker evaluates to True for the given
    python_version and sys_platform. Unconditional deps (no marker) are skipped
    because pip already handles those correctly.
    """
    deps: list[str] = []
    try:
        with zipfile.ZipFile(whl_path) as zf:
            metadata_names = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
            if not metadata_names:
                return []
            metadata = zf.read(metadata_names[0]).decode("utf-8", errors="replace")
            for line in metadata.splitlines():
                if not line.lower().startswith("requires-dist:"):
                    continue
                req_str = line.split(":", 1)[1].strip()
                if ";" not in req_str:
                    continue  # unconditional — pip handles these
                pkg_part, marker_part = req_str.split(";", 1)
                pkg_name = re.split(r"[\s\[><=!~(]", pkg_part.strip())[0].strip()
                if pkg_name and _eval_marker(marker_part.strip(), python_version, sys_platform):
                    deps.append(pkg_name)
    except Exception:
        pass
    return deps


def filter_stdlib(modules: set[str]) -> set[str]:
    """Remove standard library modules from a set of module names."""
    return {m for m in modules if m not in STDLIB_MODULES and not m.startswith("_")}


def map_imports_to_packages(imports: set[str]) -> set[str]:
    """Map import names to PyPI package names."""
    packages = set()
    for imp in imports:
        if imp in IMPORT_TO_PACKAGE:
            packages.add(IMPORT_TO_PACKAGE[imp])
        else:
            # Most packages have the same name as their import
            packages.add(imp)
    return packages


def plan_portable_repo(
    path: str | Path | None = None,
    add_pkgs: list[str] | None = None,
    recursive: bool = True,
    write_requirements: bool | Literal["ask"] = "ask",
    verbose: bool = True,
) -> list[str]:
    """Scan Python files to detect package dependencies.

    Parameters
    ----------
    path
        Directory to scan for Python files. If None, only `add_pkgs` are used.
    add_pkgs
        Additional packages to include in the list.
    recursive
        Whether to search subdirectories recursively.
    write_requirements
        Whether to write a requirements.txt file. Use "ask" to prompt.
    verbose
        Whether to print progress messages.

    Returns
    -------
    list[str]
        List of detected package names.
    """
    if path is None and not add_pkgs:
        raise ValueError(
            "You need to provide either a `path` with Python files "
            "or a list of packages in `add_pkgs`."
        )

    pkgs: set[str] = set(add_pkgs or [])

    if path is not None:
        path = Path(path)
        pattern_py = "**/*.py" if recursive else "*.py"
        pattern_ipynb = "**/*.ipynb" if recursive else "*.ipynb"

        # Scan Python files
        py_files = list(path.glob(pattern_py))
        if verbose:
            print(f"Scanning {len(py_files)} Python files...")

        imports: set[str] = set()
        for py_file in py_files:
            imports.update(extract_imports_from_file(py_file))

        # Scan Jupyter notebooks
        nb_files = list(path.glob(pattern_ipynb))
        if verbose:
            print(f"Scanning {len(nb_files)} Jupyter notebooks...")

        for nb_file in nb_files:
            imports.update(extract_imports_from_notebook(nb_file))

        # Filter and map
        imports = filter_stdlib(imports)
        pkgs.update(map_imports_to_packages(imports))

        if verbose:
            print(f"Found {len(pkgs)} packages (excluding standard library)")

    # Handle requirements.txt export
    if write_requirements == "ask":
        try:
            response = input("Export requirements.txt? [y/N] ").strip().lower()
            write_requirements = response in ("y", "yes")
        except EOFError:
            write_requirements = False

    if write_requirements:
        req_path = Path("requirements.txt")
        req_path.write_text("\n".join(sorted(pkgs)) + "\n")
        if verbose:
            print(f"Wrote {req_path}")

    return sorted(pkgs)


def build_portable_repo(
    pkgs: list[str],
    platform: str = "win_amd64",
    python_version: str = "3.11",
    out_file: str = "portable_repo.zip",
    include_deps: bool = True,
    include_extras: bool = True,
    verbose: bool = True,
) -> str:
    """Download packages and create a portable zip repository.

    Parameters
    ----------
    pkgs
        List of package names to download.
    platform
        Target platform string. Common values:
        - "win_amd64" (Windows 64-bit)
        - "win32" (Windows 32-bit)
        - "manylinux2014_x86_64" (Linux x86_64)
        - "manylinux2014_aarch64" (Linux ARM64)
        - "macosx_10_9_x86_64" (macOS Intel)
        - "macosx_11_0_arm64" (macOS Apple Silicon)
    python_version
        Target Python version (e.g., "3.11", "3.12").
    out_file
        Output zip file path.
    include_deps
        Whether to include dependencies (default True).
    include_extras
        Whether to also download optional extras by appending ``[all]`` to each
        package spec that doesn't already specify extras (default False).
        Packages that don't define an ``[all]`` extra will still be downloaded.
    verbose
        Whether to print progress messages.

    Returns
    -------
    str
        Path to the created zip file.
    """
    if not pkgs:
        raise ValueError("No packages specified for download.")

    download_dir = Path("portable_repo_temp")
    download_dir.mkdir(exist_ok=True)

    # Derive ABI tag from Python version so pip can find C extension wheels.
    # e.g. "3.8" -> "cp38", "3.11" -> "cp311"
    abi_tag = "cp" + python_version.replace(".", "")

    if verbose:
        print(f"Downloading {len(pkgs)} packages for {platform} / Python {python_version}...")

    # Try batch download first (fast path)
    cmd = [
        sys.executable, "-m", "pip", "download",
        "--dest", str(download_dir),
        "--platform", platform,
        "--python-version", python_version,
        "--implementation", "cp",
        "--abi", abi_tag,
        "--only-binary=:all:",
    ]

    if not include_deps:
        cmd.append("--no-deps")

    if include_extras:
        download_pkgs = [f"{p}[all]" if "[" not in p else p for p in pkgs]
    else:
        download_pkgs = pkgs

    cmd.extend(download_pkgs)

    result = subprocess.run(cmd, capture_output=True, text=True)

    failed_pkgs: list[str] = []

    if result.returncode != 0:
        # Parse stderr to find which packages failed
        # pip reports: "ERROR: No matching distribution found for <package>"
        failed_pattern = re.compile(r"No matching distribution found for ([^\s]+)")
        failed_pkgs = failed_pattern.findall(result.stderr)

        if failed_pkgs:
            # Retry without the failed packages
            failed_bases = {re.split(r"[\[><=!~]", f)[0].lower() for f in failed_pkgs}
            remaining_pkgs = [p for p in download_pkgs if re.split(r"[\[><=!~]", p)[0].lower() not in failed_bases]

            if remaining_pkgs:
                if verbose:
                    print(f"Retrying without {len(failed_pkgs)} unavailable package(s)...")

                cmd = [
                    sys.executable, "-m", "pip", "download",
                    "--dest", str(download_dir),
                    "--platform", platform,
                    "--python-version", python_version,
                    "--implementation", "cp",
                    "--abi", abi_tag,
                    "--only-binary=:all:",
                ]
                if not include_deps:
                    cmd.append("--no-deps")
                cmd.extend(remaining_pkgs)

                subprocess.run(cmd, capture_output=True, text=True)

    # Show summary of failures
    if failed_pkgs and verbose:
        print(f"\nWarning: {len(failed_pkgs)} package(s) could not be downloaded:")
        for pkg in failed_pkgs:
            print(f"  - {pkg}")
        print("These may be misidentified imports or packages without compatible wheels.\n")

    # Count downloaded files
    wheel_files = list(download_dir.glob("*.whl"))
    if verbose:
        print(f"Downloaded {len(wheel_files)} wheel files ({len(pkgs) - len(failed_pkgs)} packages succeeded, {len(failed_pkgs)} failed)")

    # Verify Requires-Python compatibility for each downloaded wheel
    incompatible_wheels: list[tuple[Path, str]] = []
    for whl in wheel_files:
        req_python = _get_wheel_requires_python(whl)
        if req_python and not _is_python_version_compatible(req_python, python_version):
            incompatible_wheels.append((whl, req_python))

    if incompatible_wheels:
        if verbose:
            print(f"\nFound {len(incompatible_wheels)} wheel(s) incompatible with Python {python_version}, searching PyPI for older compatible versions...")

        retry_specs: list[str] = []
        unresolved: list[tuple[str, str]] = []

        for whl, req in incompatible_wheels:
            pkg = whl.stem.split("-")[0].replace("_", "-").lower()
            compat_ver = _find_latest_compatible_version(pkg, python_version)
            whl.unlink()  # remove the incompatible wheel regardless
            if compat_ver:
                if verbose:
                    print(f"  {pkg}: will use =={compat_ver} (downloaded version requires: {req})")
                retry_specs.append(f"{pkg}=={compat_ver}")
            else:
                if verbose:
                    print(f"  {pkg}: no compatible version found on PyPI (latest requires: {req})")
                unresolved.append((pkg, req))

        if retry_specs:
            if verbose:
                print(f"Re-downloading {len(retry_specs)} package(s) with compatible versions (including dependencies)...")
            retry_cmd = [
                sys.executable, "-m", "pip", "download",
                "--dest", str(download_dir),
                "--platform", platform,
                "--python-version", python_version,
                "--implementation", "cp",
                "--abi", abi_tag,
                "--only-binary=:all:",
            ]
            retry_cmd.extend(retry_specs)
            subprocess.run(retry_cmd, capture_output=True, text=True)
            wheel_files = list(download_dir.glob("*.whl"))

            # Second-pass: check that the retry didn't pull in new incompatible wheels
            still_incompatible: list[tuple[Path, str]] = []
            for whl in wheel_files:
                req_python = _get_wheel_requires_python(whl)
                if req_python and not _is_python_version_compatible(req_python, python_version):
                    still_incompatible.append((whl, req_python))

            if still_incompatible:
                print(
                    f"\nWarning: {len(still_incompatible)} wheel(s) are still incompatible with "
                    f"Python {python_version} after retry and will likely cause install failures:"
                )
                for whl, req in still_incompatible:
                    print(f"  - {whl.name}  (Requires-Python: {req})")
                print()

        if unresolved:
            print(
                f"\nWarning: No Python {python_version}-compatible version found for "
                f"{len(unresolved)} package(s). These are excluded from the zip and "
                f"WILL BE MISSING at install time:"
            )
            for pkg, req in unresolved:
                print(f"  - {pkg}  (latest Requires-Python: {req})")
            print()

    # Resolve marker-conditional dependencies that pip may have missed.
    # `pip download --python-version X` evaluates environment markers (e.g.
    # `python_version < "3.9"`) against the *running* Python, not the target X.
    # We do our own pass: inspect each wheel's Requires-Dist, evaluate markers
    # for the target version ourselves, and explicitly download anything missing.
    sys_platform = _platform_to_sys_platform(platform)
    attempted_marker_pkgs: set[str] = set()
    checked_for_marker_deps: set[str] = set()

    for _iteration in range(10):  # bounded to handle transitive conditional deps
        wheel_files = list(download_dir.glob("*.whl"))
        downloaded_names = {
            whl.stem.split("-")[0].replace("_", "-").lower() for whl in wheel_files
        }
        missing_marker_deps: list[str] = []
        for whl in wheel_files:
            pkg_stem = whl.stem.split("-")[0].replace("_", "-").lower()
            if pkg_stem in checked_for_marker_deps:
                continue
            checked_for_marker_deps.add(pkg_stem)
            for dep in _get_marker_required_deps(whl, python_version, sys_platform):
                dep_norm = dep.replace("_", "-").lower()
                if dep_norm not in downloaded_names and dep_norm not in attempted_marker_pkgs:
                    missing_marker_deps.append(dep)
                    attempted_marker_pkgs.add(dep_norm)

        if not missing_marker_deps:
            break

        if verbose:
            print(
                f"Downloading {len(missing_marker_deps)} marker-conditional "
                f"dependency(ies) for Python {python_version}:"
            )
            for dep in missing_marker_deps:
                print(f"  - {dep}")

        marker_cmd = [
            sys.executable, "-m", "pip", "download",
            "--dest", str(download_dir),
            "--platform", platform,
            "--python-version", python_version,
            "--implementation", "cp",
            "--abi", abi_tag,
            "--only-binary=:all:",
            "--no-deps",  # their transitive deps are caught by subsequent iterations
        ] + missing_marker_deps
        marker_result = subprocess.run(marker_cmd, capture_output=True, text=True)

        if marker_result.returncode != 0:
            still_missing = re.compile(
                r"No matching distribution found for ([^\s]+)"
            ).findall(marker_result.stderr)
            if still_missing and verbose:
                print(
                    f"\nWarning: Could not download {len(still_missing)} "
                    f"marker-required package(s) — they will be missing at install time:"
                )
                for pkg in still_missing:
                    print(f"  - {pkg}")
                print()

    wheel_files = list(download_dir.glob("*.whl"))

    # Create zip archive
    if verbose:
        print(f"Creating {out_file}...")

    with zipfile.ZipFile(out_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for whl in wheel_files:
            zf.write(whl, whl.name)

    # Clean up temp directory
    for f in download_dir.iterdir():
        f.unlink()
    download_dir.rmdir()

    if verbose:
        print(f"Created {out_file} with {len(wheel_files)} packages")

    return out_file


def export_install_script(
    requirements_file: str = "requirements.txt",
    wheels_dir: str = "wheels",
    script_file: str = "install_offline.sh",
    verbose: bool = True,
) -> str:
    """Generate a shell script for offline installation.

    Parameters
    ----------
    requirements_file
        Path to requirements.txt file.
    wheels_dir
        Directory where wheels will be extracted.
    script_file
        Output script file path.
    verbose
        Whether to print progress messages.

    Returns
    -------
    str
        Path to the created script file.
    """
    script = f"""#!/bin/bash
# Offline installation script generated by sane-gesis
#
# Usage:
#   1. Extract portable_repo.zip to {wheels_dir}/
#   2. Run this script: bash {script_file}

set -e

WHEELS_DIR="{wheels_dir}"
REQUIREMENTS="{requirements_file}"

if [ ! -d "$WHEELS_DIR" ]; then
    echo "Error: Wheels directory '$WHEELS_DIR' not found."
    echo "Please extract portable_repo.zip first:"
    echo "  unzip portable_repo.zip -d $WHEELS_DIR"
    exit 1
fi

if [ ! -f "$REQUIREMENTS" ]; then
    echo "Error: Requirements file '$REQUIREMENTS' not found."
    exit 1
fi

echo "Installing packages from $WHEELS_DIR..."
pip install --no-index --find-links="$WHEELS_DIR" -r "$REQUIREMENTS"

echo "Installation complete!"
"""

    Path(script_file).write_text(script)

    if verbose:
        print(f"Created installation script: {script_file}")

    return script_file
