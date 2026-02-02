"""Build a Portable Package Repository from Python Project Files.

Scans Python scripts and Jupyter notebooks in a directory to detect package
dependencies, downloads all required packages and their dependencies, and
creates a compressed portable package repository.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import zipfile
from pathlib import Path
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

    if verbose:
        print(f"Downloading {len(pkgs)} packages for {platform} / Python {python_version}...")

    # Build pip download command
    cmd = [
        sys.executable, "-m", "pip", "download",
        "--dest", str(download_dir),
        "--platform", platform,
        "--python-version", python_version,
        "--only-binary=:all:",
    ]

    if not include_deps:
        cmd.append("--no-deps")

    cmd.extend(pkgs)

    # Run pip download
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Check for packages that failed
        if verbose:
            print("Warning: Some packages may have failed to download.")
            print(result.stderr)

    # Count downloaded files
    wheel_files = list(download_dir.glob("*.whl"))
    if verbose:
        print(f"Downloaded {len(wheel_files)} wheel files")

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
