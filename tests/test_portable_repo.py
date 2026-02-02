"""Tests for sane_gesis.portable_repo module."""

import tempfile
from pathlib import Path

import pytest

from sane_gesis.portable_repo import (
    extract_imports_from_source,
    extract_imports_from_file,
    extract_imports_from_notebook,
    filter_stdlib,
    map_imports_to_packages,
    plan_portable_repo,
)


class TestExtractImports:
    """Tests for import extraction functions."""

    def test_extract_simple_import(self):
        source = "import numpy"
        imports = extract_imports_from_source(source)
        assert "numpy" in imports

    def test_extract_from_import(self):
        source = "from pandas import DataFrame"
        imports = extract_imports_from_source(source)
        assert "pandas" in imports

    def test_extract_multiple_imports(self):
        source = """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
"""
        imports = extract_imports_from_source(source)
        assert "numpy" in imports
        assert "pandas" in imports
        assert "sklearn" in imports
        assert "matplotlib" in imports

    def test_extract_submodule_import(self):
        source = "import os.path"
        imports = extract_imports_from_source(source)
        assert "os" in imports
        assert "os.path" not in imports

    def test_extract_handles_syntax_error(self):
        source = "import numpy\nthis is not valid python {"
        imports = extract_imports_from_source(source)
        # Should return empty set on syntax error
        assert imports == set()

    def test_extract_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("import requests\nfrom bs4 import BeautifulSoup")
            f.flush()
            imports = extract_imports_from_file(Path(f.name))
            assert "requests" in imports
            assert "bs4" in imports


class TestExtractNotebook:
    """Tests for Jupyter notebook extraction."""

    def test_extract_from_notebook(self):
        notebook_content = """{
  "cells": [
    {
      "cell_type": "code",
      "source": ["import pandas as pd\\n", "import numpy as np"]
    },
    {
      "cell_type": "markdown",
      "source": ["# This is markdown"]
    },
    {
      "cell_type": "code",
      "source": ["from sklearn import linear_model"]
    }
  ],
  "metadata": {},
  "nbformat": 4,
  "nbformat_minor": 5
}"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
            f.write(notebook_content)
            f.flush()
            imports = extract_imports_from_notebook(Path(f.name))
            assert "pandas" in imports
            assert "numpy" in imports
            assert "sklearn" in imports

    def test_extract_from_invalid_notebook(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
            f.write("not valid json")
            f.flush()
            imports = extract_imports_from_notebook(Path(f.name))
            assert imports == set()


class TestFilterStdlib:
    """Tests for standard library filtering."""

    def test_filter_removes_stdlib(self):
        modules = {"os", "sys", "json", "numpy", "pandas"}
        filtered = filter_stdlib(modules)
        assert "os" not in filtered
        assert "sys" not in filtered
        assert "json" not in filtered
        assert "numpy" in filtered
        assert "pandas" in filtered

    def test_filter_removes_private_modules(self):
        modules = {"_thread", "_collections", "numpy"}
        filtered = filter_stdlib(modules)
        assert "_thread" not in filtered
        assert "_collections" not in filtered
        assert "numpy" in filtered


class TestMapImportsToPackages:
    """Tests for import-to-package mapping."""

    def test_maps_cv2_to_opencv(self):
        imports = {"cv2"}
        packages = map_imports_to_packages(imports)
        assert "opencv-python" in packages
        assert "cv2" not in packages

    def test_maps_pil_to_pillow(self):
        imports = {"PIL"}
        packages = map_imports_to_packages(imports)
        assert "pillow" in packages

    def test_maps_sklearn_to_scikit_learn(self):
        imports = {"sklearn"}
        packages = map_imports_to_packages(imports)
        assert "scikit-learn" in packages

    def test_keeps_unknown_packages(self):
        imports = {"numpy", "pandas", "some_unknown_pkg"}
        packages = map_imports_to_packages(imports)
        assert "numpy" in packages
        assert "pandas" in packages
        assert "some_unknown_pkg" in packages


class TestPlanPortableRepo:
    """Tests for plan_portable_repo function."""

    def test_plan_with_add_pkgs_only(self):
        pkgs = plan_portable_repo(
            path=None,
            add_pkgs=["numpy", "pandas"],
            write_requirements=False,
            verbose=False,
        )
        assert "numpy" in pkgs
        assert "pandas" in pkgs

    def test_plan_raises_without_args(self):
        with pytest.raises(ValueError, match="need to provide"):
            plan_portable_repo(path=None, add_pkgs=None, verbose=False)

    def test_plan_scans_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test Python file
            py_file = Path(tmpdir) / "test_script.py"
            py_file.write_text("import requests\nfrom bs4 import BeautifulSoup")

            pkgs = plan_portable_repo(
                path=tmpdir,
                write_requirements=False,
                verbose=False,
            )
            assert "requests" in pkgs
            assert "beautifulsoup4" in pkgs  # bs4 maps to beautifulsoup4
