# sane.gesis.python - Portable Package Repository Builder

A Python equivalent of the R `portable_repo` tool for creating offline-installable package repositories.

## Overview

This tool helps users working on secure offline machines by:
1. Scanning Python projects to detect package dependencies
2. Downloading package wheels (binaries) for specific platforms
3. Creating a portable zip file that can be transferred to offline machines

## Package Structure

```
sane.gesis.python/
├── pyproject.toml          # Package metadata and dependencies
├── README.md               # Usage documentation
├── src/
│   └── sane_gesis/
│       ├── __init__.py
│       └── portable_repo.py    # Main module
└── tests/
    └── test_portable_repo.py   # Unit tests
```

## Implementation Steps

### Step 1: Set up package scaffolding
- [x] Create `pyproject.toml` with package metadata
- [x] Create `src/sane_gesis/__init__.py`
- [x] Create basic `README.md`

### Step 2: Implement `plan_portable_repo()`
- [x] Scan `.py` files using AST parsing to extract imports
- [x] Scan Jupyter notebooks (`.ipynb`) for imports
- [x] Filter out standard library modules
- [x] Handle import name → package name mapping (e.g., `cv2` → `opencv-python`)
- [x] Optional: write `requirements.txt`

### Step 3: Implement `build_portable_repo()`
- [x] Use `pip download` to fetch wheels for target platform
- [x] Support platform targeting (Windows, Linux, macOS)
- [x] Support Python version targeting
- [x] Handle download failures gracefully
- [x] Create compressed zip archive

### Step 4: Add CLI interface
- [x] Create command-line entry point using `argparse`
- [x] Support `plan`, `build`, and `export-script` subcommands

### Step 5: Add helper utilities
- [x] `export_install_script()` - Generate installation script for offline machine
- [x] Standard library detection (using `sys.stdlib_module_names`)

## Key Dependencies

- `packaging` - For version parsing and platform tags
- `pip` - Used via subprocess for downloading (most reliable)
- No heavy dependencies to keep the tool portable itself

## Platform Strings for `pip download`

| Target OS | Platform string |
|-----------|-----------------|
| Windows 64-bit | `win_amd64` |
| Windows 32-bit | `win32` |
| Linux (glibc) | `manylinux2014_x86_64` |
| Linux ARM | `manylinux2014_aarch64` |
| macOS Intel | `macosx_10_9_x86_64` |
| macOS ARM | `macosx_11_0_arm64` |

## Usage Example

```python
from sane_gesis import plan_portable_repo, build_portable_repo

# Scan project for dependencies
pkgs = plan_portable_repo(path="./my_project", write_requirements=True)

# Download for Windows
build_portable_repo(pkgs, platform="win_amd64", python_version="3.11")
```

```bash
# Or via CLI
sane-gesis plan ./my_project --write-requirements
sane-gesis build -r requirements.txt --platform win_amd64 --python-version 3.11
```

## On the Offline Machine

```bash
unzip portable_repo.zip -d wheels/
pip install --no-index --find-links=wheels/ -r requirements.txt
```
