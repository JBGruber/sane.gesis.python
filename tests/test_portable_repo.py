"""Tests for sane_gesis.portable_repo module."""

import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sane_gesis.portable_repo import (
    _eval_marker,
    _get_marker_required_deps,
    _get_wheel_requires_python,
    _is_python_version_compatible,
    _platform_to_sys_platform,
    build_portable_repo,
    export_install_script,
    extract_imports_from_file,
    extract_imports_from_notebook,
    extract_imports_from_source,
    filter_stdlib,
    map_imports_to_packages,
    plan_portable_repo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wheel(tmp_path: Path, name: str = "mypackage", version: str = "1.0", metadata: str = "") -> Path:
    """Create a minimal fake wheel (.whl) zip file with a METADATA entry."""
    whl_path = tmp_path / f"{name}-{version}-py3-none-any.whl"
    dist_info = f"{name}-{version}.dist-info/METADATA"
    default_meta = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    with zipfile.ZipFile(whl_path, "w") as zf:
        zf.writestr(dist_info, default_meta + metadata)
    return whl_path


# ---------------------------------------------------------------------------
# extract_imports_from_source
# ---------------------------------------------------------------------------

class TestExtractImportsFromSource:
    def test_simple_import(self):
        imports = extract_imports_from_source("import numpy")
        assert "numpy" in imports

    def test_from_import(self):
        imports = extract_imports_from_source("from pandas import DataFrame")
        assert "pandas" in imports

    def test_multiple_imports(self):
        source = """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
"""
        imports = extract_imports_from_source(source)
        assert {"numpy", "pandas", "sklearn", "matplotlib"} <= imports

    def test_submodule_import_uses_top_level(self):
        imports = extract_imports_from_source("import os.path")
        assert "os" in imports
        assert "os.path" not in imports

    def test_from_submodule_import_uses_top_level(self):
        imports = extract_imports_from_source("from xml.etree import ElementTree")
        assert "xml" in imports

    def test_relative_imports_excluded(self):
        # Relative imports (level > 0) should not be included
        imports = extract_imports_from_source("from . import utils\nfrom ..helpers import foo")
        assert "utils" not in imports
        assert "helpers" not in imports

    def test_syntax_error_returns_empty(self):
        imports = extract_imports_from_source("import numpy\nthis is not valid python {")
        assert imports == set()

    def test_empty_source_returns_empty(self):
        assert extract_imports_from_source("") == set()

    def test_no_imports_returns_empty(self):
        assert extract_imports_from_source("x = 1 + 2") == set()

    def test_multiline_imports(self):
        source = "import os, sys, re"
        imports = extract_imports_from_source(source)
        assert {"os", "sys", "re"} <= imports


# ---------------------------------------------------------------------------
# extract_imports_from_file
# ---------------------------------------------------------------------------

class TestExtractImportsFromFile:
    def test_reads_python_file(self, tmp_path):
        py_file = tmp_path / "script.py"
        py_file.write_text("import requests\nfrom bs4 import BeautifulSoup")
        imports = extract_imports_from_file(py_file)
        assert "requests" in imports
        assert "bs4" in imports

    def test_nonexistent_file_returns_empty(self, tmp_path):
        result = extract_imports_from_file(tmp_path / "does_not_exist.py")
        assert result == set()

    def test_unicode_decode_error_returns_empty(self, tmp_path):
        bad_file = tmp_path / "bad.py"
        bad_file.write_bytes(b"\xff\xfe import something")
        # May succeed or fail depending on the bytes; if it fails it should return empty
        result = extract_imports_from_file(bad_file)
        assert isinstance(result, set)


# ---------------------------------------------------------------------------
# extract_imports_from_notebook
# ---------------------------------------------------------------------------

class TestExtractImportsFromNotebook:
    def _write_notebook(self, tmp_path, cells):
        nb = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
        nb_file = tmp_path / "notebook.ipynb"
        nb_file.write_text(json.dumps(nb))
        return nb_file

    def test_extracts_from_code_cells(self, tmp_path):
        nb_file = self._write_notebook(tmp_path, [
            {"cell_type": "code", "source": ["import pandas as pd\n", "import numpy as np"]},
            {"cell_type": "code", "source": ["from sklearn import linear_model"]},
        ])
        imports = extract_imports_from_notebook(nb_file)
        assert {"pandas", "numpy", "sklearn"} <= imports

    def test_ignores_markdown_cells(self, tmp_path):
        nb_file = self._write_notebook(tmp_path, [
            {"cell_type": "markdown", "source": ["import os"]},
        ])
        imports = extract_imports_from_notebook(nb_file)
        assert "os" not in imports

    def test_empty_cells_list(self, tmp_path):
        nb_file = self._write_notebook(tmp_path, [])
        assert extract_imports_from_notebook(nb_file) == set()

    def test_missing_cells_key(self, tmp_path):
        nb_file = tmp_path / "notebook.ipynb"
        nb_file.write_text(json.dumps({"metadata": {}}))
        assert extract_imports_from_notebook(nb_file) == set()

    def test_invalid_json_returns_empty(self, tmp_path):
        nb_file = tmp_path / "bad.ipynb"
        nb_file.write_text("not valid json")
        assert extract_imports_from_notebook(nb_file) == set()

    def test_nonexistent_file_returns_empty(self, tmp_path):
        assert extract_imports_from_notebook(tmp_path / "missing.ipynb") == set()


# ---------------------------------------------------------------------------
# filter_stdlib
# ---------------------------------------------------------------------------

class TestFilterStdlib:
    def test_removes_stdlib_modules(self):
        filtered = filter_stdlib({"os", "sys", "json", "numpy", "pandas"})
        assert "os" not in filtered
        assert "sys" not in filtered
        assert "json" not in filtered
        assert {"numpy", "pandas"} <= filtered

    def test_removes_private_modules(self):
        filtered = filter_stdlib({"_thread", "_collections", "numpy"})
        assert "_thread" not in filtered
        assert "_collections" not in filtered
        assert "numpy" in filtered

    def test_empty_set(self):
        assert filter_stdlib(set()) == set()

    def test_all_stdlib_returns_empty(self):
        assert filter_stdlib({"os", "sys", "re", "json"}) == set()

    def test_no_stdlib_unchanged(self):
        pkgs = {"numpy", "pandas", "requests"}
        assert filter_stdlib(pkgs) == pkgs


# ---------------------------------------------------------------------------
# map_imports_to_packages
# ---------------------------------------------------------------------------

class TestMapImportsToPackages:
    def test_cv2_to_opencv(self):
        assert "opencv-python" in map_imports_to_packages({"cv2"})

    def test_pil_to_pillow(self):
        assert "pillow" in map_imports_to_packages({"PIL"})

    def test_sklearn_to_scikit_learn(self):
        assert "scikit-learn" in map_imports_to_packages({"sklearn"})

    def test_bs4_to_beautifulsoup4(self):
        assert "beautifulsoup4" in map_imports_to_packages({"bs4"})

    def test_yaml_to_pyyaml(self):
        assert "pyyaml" in map_imports_to_packages({"yaml"})

    def test_unknown_import_kept_as_is(self):
        pkgs = map_imports_to_packages({"some_unknown_pkg"})
        assert "some_unknown_pkg" in pkgs

    def test_known_imports_not_kept_raw(self):
        pkgs = map_imports_to_packages({"cv2"})
        assert "cv2" not in pkgs

    def test_empty_set(self):
        assert map_imports_to_packages(set()) == set()

    def test_mixed_known_and_unknown(self):
        pkgs = map_imports_to_packages({"cv2", "numpy", "bs4"})
        assert "opencv-python" in pkgs
        assert "numpy" in pkgs
        assert "beautifulsoup4" in pkgs


# ---------------------------------------------------------------------------
# _is_python_version_compatible
# ---------------------------------------------------------------------------

class TestIsPythonVersionCompatible:
    def test_gte_satisfied(self):
        assert _is_python_version_compatible(">=3.8", "3.9.0")

    def test_gte_not_satisfied(self):
        assert not _is_python_version_compatible(">=3.8", "3.7.0")

    def test_lte_satisfied(self):
        assert _is_python_version_compatible("<=3.12", "3.11.0")

    def test_lte_not_satisfied(self):
        assert not _is_python_version_compatible("<=3.10", "3.11.0")

    def test_lt_satisfied(self):
        assert _is_python_version_compatible("<3.9", "3.8.0")

    def test_lt_not_satisfied(self):
        assert not _is_python_version_compatible("<3.9", "3.9.0")

    def test_gt_satisfied(self):
        assert _is_python_version_compatible(">3.8", "3.9.0")

    def test_gt_not_satisfied(self):
        assert not _is_python_version_compatible(">3.9", "3.9.0")

    def test_eq_satisfied(self):
        assert _is_python_version_compatible("==3.8", "3.8.0")

    def test_eq_not_satisfied(self):
        assert not _is_python_version_compatible("==3.8", "3.9.0")

    def test_ne_satisfied(self):
        assert _is_python_version_compatible("!=3.9", "3.10.0")

    def test_ne_not_satisfied(self):
        assert not _is_python_version_compatible("!=3.9", "3.9.0")

    def test_compatible_release_satisfied(self):
        # ~=3.8 means >=3.8, ==3.*
        assert _is_python_version_compatible("~=3.8", "3.9.0")

    def test_compatible_release_not_satisfied(self):
        assert not _is_python_version_compatible("~=3.8", "4.0.0")

    def test_combined_specifiers(self):
        assert _is_python_version_compatible(">=3.8,<4.0", "3.11.0")
        assert not _is_python_version_compatible(">=3.8,<3.10", "3.11.0")

    def test_exact_boundary_gte(self):
        assert _is_python_version_compatible(">=3.8", "3.8.0")

    def test_exact_boundary_lte(self):
        assert _is_python_version_compatible("<=3.8", "3.8.0")


# ---------------------------------------------------------------------------
# _platform_to_sys_platform
# ---------------------------------------------------------------------------

class TestPlatformToSysPlatform:
    def test_win_amd64(self):
        assert _platform_to_sys_platform("win_amd64") == "win32"

    def test_win32(self):
        assert _platform_to_sys_platform("win32") == "win32"

    def test_linux(self):
        assert _platform_to_sys_platform("manylinux2014_x86_64") == "linux"

    def test_linux_aarch64(self):
        assert _platform_to_sys_platform("manylinux2014_aarch64") == "linux"

    def test_macos_intel(self):
        assert _platform_to_sys_platform("macosx_10_9_x86_64") == "darwin"

    def test_macos_arm(self):
        assert _platform_to_sys_platform("macosx_11_0_arm64") == "darwin"

    def test_darwin_string(self):
        assert _platform_to_sys_platform("darwin") == "darwin"

    def test_unknown_defaults_to_linux(self):
        assert _platform_to_sys_platform("unknown_platform") == "linux"


# ---------------------------------------------------------------------------
# _eval_marker
# ---------------------------------------------------------------------------

class TestEvalMarker:
    def test_python_version_gte_true(self):
        assert _eval_marker('python_version >= "3.8"', "3.9", "linux")

    def test_python_version_gte_false(self):
        assert not _eval_marker('python_version >= "3.10"', "3.9", "linux")

    def test_python_version_lt_true(self):
        assert _eval_marker('python_version < "3.9"', "3.8", "linux")

    def test_python_version_lt_false(self):
        assert not _eval_marker('python_version < "3.9"', "3.9", "linux")

    def test_sys_platform_win32_true(self):
        assert _eval_marker('sys_platform == "win32"', "3.9", "win32")

    def test_sys_platform_win32_false(self):
        assert not _eval_marker('sys_platform == "win32"', "3.9", "linux")

    def test_sys_platform_linux_true(self):
        assert _eval_marker('sys_platform == "linux"', "3.9", "linux")

    def test_os_name_nt_windows(self):
        assert _eval_marker('os_name == "nt"', "3.9", "win32")

    def test_os_name_posix_linux(self):
        assert _eval_marker('os_name == "posix"', "3.9", "linux")

    def test_unparseable_marker_returns_true(self):
        # Conservative: assume required if marker can't be parsed
        assert _eval_marker("some_unknown_marker == 'value'", "3.9", "linux")


# ---------------------------------------------------------------------------
# _get_wheel_requires_python
# ---------------------------------------------------------------------------

class TestGetWheelRequiresPython:
    def test_extracts_requires_python(self, tmp_path):
        whl = _make_wheel(tmp_path, metadata="Requires-Python: >=3.8\n")
        assert _get_wheel_requires_python(whl) == ">=3.8"

    def test_returns_none_when_absent(self, tmp_path):
        whl = _make_wheel(tmp_path)
        assert _get_wheel_requires_python(whl) is None

    def test_returns_none_for_missing_dist_info(self, tmp_path):
        whl_path = tmp_path / "pkg-1.0-py3-none-any.whl"
        with zipfile.ZipFile(whl_path, "w") as zf:
            zf.writestr("some_other_file.txt", "content")
        assert _get_wheel_requires_python(whl_path) is None

    def test_returns_none_for_invalid_zip(self, tmp_path):
        bad = tmp_path / "bad.whl"
        bad.write_text("not a zip file")
        assert _get_wheel_requires_python(bad) is None

    def test_case_insensitive_field_name(self, tmp_path):
        whl = _make_wheel(tmp_path, metadata="REQUIRES-PYTHON: >=3.9\n")
        result = _get_wheel_requires_python(whl)
        assert result == ">=3.9"


# ---------------------------------------------------------------------------
# _get_marker_required_deps
# ---------------------------------------------------------------------------

class TestGetMarkerRequiredDeps:
    def test_returns_conditional_dep_when_marker_true(self, tmp_path):
        metadata = (
            "Requires-Dist: typing-extensions; python_version < \"3.11\"\n"
        )
        whl = _make_wheel(tmp_path, metadata=metadata)
        deps = _get_marker_required_deps(whl, "3.10", "linux")
        assert "typing-extensions" in deps

    def test_excludes_conditional_dep_when_marker_false(self, tmp_path):
        metadata = (
            "Requires-Dist: typing-extensions; python_version < \"3.11\"\n"
        )
        whl = _make_wheel(tmp_path, metadata=metadata)
        deps = _get_marker_required_deps(whl, "3.12", "linux")
        assert "typing-extensions" not in deps

    def test_excludes_unconditional_deps(self, tmp_path):
        metadata = "Requires-Dist: requests\n"
        whl = _make_wheel(tmp_path, metadata=metadata)
        deps = _get_marker_required_deps(whl, "3.11", "linux")
        assert "requests" not in deps

    def test_platform_conditional_dep_included(self, tmp_path):
        metadata = (
            'Requires-Dist: pywin32; sys_platform == "win32"\n'
        )
        whl = _make_wheel(tmp_path, metadata=metadata)
        assert "pywin32" in _get_marker_required_deps(whl, "3.11", "win32")
        assert "pywin32" not in _get_marker_required_deps(whl, "3.11", "linux")

    def test_invalid_zip_returns_empty(self, tmp_path):
        bad = tmp_path / "bad.whl"
        bad.write_text("not a zip")
        assert _get_marker_required_deps(bad, "3.11", "linux") == []

    def test_no_dist_info_returns_empty(self, tmp_path):
        whl_path = tmp_path / "pkg-1.0-py3-none-any.whl"
        with zipfile.ZipFile(whl_path, "w") as zf:
            zf.writestr("other.txt", "")
        assert _get_marker_required_deps(whl_path, "3.11", "linux") == []


# ---------------------------------------------------------------------------
# export_install_script
# ---------------------------------------------------------------------------

class TestExportInstallScript:
    def test_creates_script_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = export_install_script(verbose=False)
        assert Path(result).exists()

    def test_returns_script_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = export_install_script(script_file="install.sh", verbose=False)
        assert result == "install.sh"

    def test_script_contains_wheels_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        export_install_script(wheels_dir="my_wheels", verbose=False)
        content = Path("install_offline.sh").read_text()
        assert "my_wheels" in content

    def test_script_contains_requirements_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        export_install_script(requirements_file="my_reqs.txt", verbose=False)
        content = Path("install_offline.sh").read_text()
        assert "my_reqs.txt" in content

    def test_script_has_pip_install_command(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        export_install_script(verbose=False)
        content = Path("install_offline.sh").read_text()
        assert "pip install" in content
        assert "--no-index" in content
        assert "--find-links" in content

    def test_custom_script_file_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = export_install_script(script_file="custom_install.sh", verbose=False)
        assert Path("custom_install.sh").exists()
        assert result == "custom_install.sh"


# ---------------------------------------------------------------------------
# plan_portable_repo
# ---------------------------------------------------------------------------

class TestPlanPortableRepo:
    def test_add_pkgs_only(self):
        pkgs = plan_portable_repo(add_pkgs=["numpy", "pandas"], write_requirements=False, verbose=False)
        assert "numpy" in pkgs
        assert "pandas" in pkgs

    def test_raises_without_path_or_pkgs(self):
        with pytest.raises(ValueError, match="need to provide"):
            plan_portable_repo(path=None, add_pkgs=None, verbose=False)

    def test_scans_python_files(self, tmp_path):
        (tmp_path / "script.py").write_text("import requests\nfrom bs4 import BeautifulSoup")
        pkgs = plan_portable_repo(path=tmp_path, write_requirements=False, verbose=False)
        assert "requests" in pkgs
        assert "beautifulsoup4" in pkgs

    def test_scans_notebooks(self, tmp_path):
        nb = {
            "cells": [{"cell_type": "code", "source": ["import plotly"]}],
            "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
        }
        (tmp_path / "analysis.ipynb").write_text(json.dumps(nb))
        pkgs = plan_portable_repo(path=tmp_path, write_requirements=False, verbose=False)
        assert "plotly" in pkgs

    def test_combines_path_and_add_pkgs(self, tmp_path):
        (tmp_path / "script.py").write_text("import requests")
        pkgs = plan_portable_repo(path=tmp_path, add_pkgs=["extra-pkg"], write_requirements=False, verbose=False)
        assert "requests" in pkgs
        assert "extra-pkg" in pkgs

    def test_excludes_stdlib(self, tmp_path):
        (tmp_path / "script.py").write_text("import os\nimport sys\nimport json")
        pkgs = plan_portable_repo(path=tmp_path, write_requirements=False, verbose=False)
        assert "os" not in pkgs
        assert "sys" not in pkgs
        assert "json" not in pkgs

    def test_returns_sorted_list(self):
        pkgs = plan_portable_repo(add_pkgs=["zebra", "apple", "mango"], write_requirements=False, verbose=False)
        assert pkgs == sorted(pkgs)

    def test_non_recursive_scan(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.py").write_text("import numpy")
        pkgs = plan_portable_repo(path=tmp_path, recursive=False, write_requirements=False, verbose=False)
        assert "numpy" not in pkgs

    def test_recursive_scan(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.py").write_text("import numpy")
        pkgs = plan_portable_repo(path=tmp_path, recursive=True, write_requirements=False, verbose=False)
        assert "numpy" in pkgs

    def test_writes_requirements_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        plan_portable_repo(add_pkgs=["numpy", "pandas"], write_requirements=True, verbose=False)
        req = Path("requirements.txt").read_text()
        assert "numpy" in req
        assert "pandas" in req


# ---------------------------------------------------------------------------
# build_portable_repo
# ---------------------------------------------------------------------------

class TestBuildPortableRepo:
    def test_raises_for_empty_pkgs(self):
        with pytest.raises(ValueError, match="No packages specified"):
            build_portable_repo([])

    def test_calls_pip_download(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with patch("sane_gesis.portable_repo.subprocess.run", return_value=mock_result) as mock_run:
            build_portable_repo(["numpy"], out_file=str(tmp_path / "out.zip"), verbose=False)

        assert mock_run.called
        cmd_args = mock_run.call_args_list[0][0][0]
        assert "pip" in cmd_args
        assert "download" in cmd_args
        assert any("numpy" in arg for arg in cmd_args)

    def test_includes_platform_and_python_version(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mock_result = MagicMock(returncode=0, stderr="")

        with patch("sane_gesis.portable_repo.subprocess.run", return_value=mock_result) as mock_run:
            build_portable_repo(
                ["numpy"],
                platform="win_amd64",
                python_version="3.11",
                out_file=str(tmp_path / "out.zip"),
                verbose=False,
            )

        cmd_args = mock_run.call_args_list[0][0][0]
        assert "--platform" in cmd_args
        assert "win_amd64" in cmd_args
        assert "--python-version" in cmd_args
        assert "3.11" in cmd_args

    def test_no_deps_flag(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mock_result = MagicMock(returncode=0, stderr="")

        with patch("sane_gesis.portable_repo.subprocess.run", return_value=mock_result) as mock_run:
            build_portable_repo(
                ["numpy"],
                include_deps=False,
                out_file=str(tmp_path / "out.zip"),
                verbose=False,
            )

        cmd_args = mock_run.call_args_list[0][0][0]
        assert "--no-deps" in cmd_args

    def test_include_extras_adds_all(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mock_result = MagicMock(returncode=0, stderr="")

        with patch("sane_gesis.portable_repo.subprocess.run", return_value=mock_result) as mock_run:
            build_portable_repo(
                ["numpy"],
                include_extras=True,
                out_file=str(tmp_path / "out.zip"),
                verbose=False,
            )

        cmd_args = mock_run.call_args_list[0][0][0]
        assert "numpy[all]" in cmd_args

    def test_no_extras_does_not_add_all(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mock_result = MagicMock(returncode=0, stderr="")

        with patch("sane_gesis.portable_repo.subprocess.run", return_value=mock_result) as mock_run:
            build_portable_repo(
                ["numpy"],
                include_extras=False,
                out_file=str(tmp_path / "out.zip"),
                verbose=False,
            )

        cmd_args = mock_run.call_args_list[0][0][0]
        assert "numpy[all]" not in cmd_args
        assert "numpy" in cmd_args

    def test_creates_output_zip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mock_result = MagicMock(returncode=0, stderr="")

        out_file = str(tmp_path / "my_repo.zip")
        with patch("sane_gesis.portable_repo.subprocess.run", return_value=mock_result):
            result = build_portable_repo(["numpy"], out_file=out_file, verbose=False)

        assert result == out_file
        # zip is created even if no wheels were downloaded (empty archive)
        assert Path(out_file).exists()

    def test_package_with_existing_extras_not_doubled(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mock_result = MagicMock(returncode=0, stderr="")

        with patch("sane_gesis.portable_repo.subprocess.run", return_value=mock_result) as mock_run:
            build_portable_repo(
                ["numpy[optional]"],
                include_extras=True,
                out_file=str(tmp_path / "out.zip"),
                verbose=False,
            )

        cmd_args = mock_run.call_args_list[0][0][0]
        # Should NOT add [all] since [optional] is already specified
        assert "numpy[optional]" in cmd_args
        assert "numpy[optional][all]" not in cmd_args
