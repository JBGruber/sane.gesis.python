# sane-gesis

> [!WARNING]  
> Entirely vibe-coded translation of the R version: https://github.com/JBGruber/sane.gesis/

Build portable Python package repositories for offline installation.

This tool helps users working on secure offline machines by scanning Python projects to detect dependencies, downloading package wheels for specific platforms, and creating portable zip archives.

## Installation

```bash
uv pip install git+https://github.com/JBGruber/sane.gesis.python.git
```

Or install from source:

```bash
uv venv 
source .venv/bin/activate
uv pip install -e .
```

## Usage

### 1. Scan your project for dependencies

```bash
# Scan current directory
sane-gesis plan .

# Scan with additional packages
sane-gesis plan ./my_project --add numpy pandas matplotlib

# Write requirements.txt automatically
sane-gesis plan . --write-requirements
```

### 2. Download packages for offline use

```bash
# Download packages for Windows
sane-gesis build -r requirements.txt --platform win_amd64 --python-version 3.8

# Download for Linux
sane-gesis build -r requirements.txt --platform manylinux2014_x86_64

# Download specific packages
sane-gesis build -p numpy pandas scikit-learn --platform win_amd64
```

### 3. Generate installation script

```bash
sane-gesis export-script
```

### On the offline machine

```bash
# Extract the portable repository
unzip portable_repo.zip -d wheels/

# Install packages
pip install --no-index --find-links=wheels/ -r requirements.txt

# Or use the generated script
bash install_offline.sh
```

## Python API

```python
from sane_gesis import plan_portable_repo, build_portable_repo

# Detect dependencies
pkgs = plan_portable_repo(
    path="./my_project",
    add_pkgs=["extra-package"],
    write_requirements=True
)

# Download for target platform
build_portable_repo(
    pkgs,
    platform="win_amd64",
    python_version="3.11",
    out_file="portable_repo.zip"
)
```

## Platform Strings

| Target OS | Platform string |
|-----------|-----------------|
| Windows 64-bit | `win_amd64` |
| Windows 32-bit | `win32` |
| Linux x86_64 (glibc) | `manylinux2014_x86_64` |
| Linux ARM64 | `manylinux2014_aarch64` |
| macOS Intel | `macosx_10_9_x86_64` |
| macOS Apple Silicon | `macosx_11_0_arm64` |

## Limitations

- Only downloads wheel files (binary packages). Packages without wheels for the target platform will fail.
- Import name detection may not catch all packages (e.g., plugins loaded dynamically).
- Some import names differ from package names; common mappings are included but may not cover all cases.

## License

MIT
