# Installation Guide

ARC-SCOPE uses optional dependency groups so you only install what you need.

## Prerequisites

- Python >= 3.9
- pip >= 21.0

## Core Package

The core package installs numpy, xarray, scipy, and pandas. It provides the bridge module for converting ARC outputs to SCOPE format, plus utilities for geometry and I/O.

```bash
pip install arc-scope
```

## Recommended First Run

The showcase experiment runs on the core package only. It already includes a bundled local weather file, so you do not need the ERA5 extra for the first end-to-end walkthrough.

```bash
pip install arc-scope
python3 -m arc_scope.experiments.showcase --output-dir ./showcase-output
```

This path exercises the verified in-repo core stack before you add ARC, SCOPE, or ERA5 dependencies.

## With ARC Support

ARC retrieves biophysical parameters from Sentinel-2 satellite imagery. It requires GDAL, which must be installed at the system level before installing the Python bindings.

### Install GDAL

**Ubuntu / Debian:**

```bash
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
```

**macOS (Homebrew):**

```bash
brew install gdal
```

**Conda (any platform):**

```bash
conda install -c conda-forge gdal
```

### Install ARC-SCOPE with ARC

```bash
pip install "arc-scope[arc]"
```

This installs ARC directly from its GitHub repository. A working GDAL installation is required for the build to succeed.

## With SCOPE Support

SCOPE runs radiative-transfer simulations on a PyTorch backend. This group installs `scope-rtm` and `torch`.

```bash
pip install "arc-scope[scope]"
```

If you need GPU support, install PyTorch for your CUDA version first, then install the SCOPE group:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "arc-scope[scope]"
```

## With Weather Data (ERA5)

The weather module uses the Copernicus Climate Data Store API to download ERA5 reanalysis data.

```bash
pip install "arc-scope[weather]"
```

### ERA5 Credentials

You need a free CDS account and API key:

1. Register at https://cds.climate.copernicus.eu/
2. Log in and visit https://cds.climate.copernicus.eu/api-how-to
3. Create `~/.cdsapirc` with your credentials:

```
url: https://cds.climate.copernicus.eu/api/v2
key: <UID>:<API-KEY>
```

Replace `<UID>` and `<API-KEY>` with the values from your CDS profile page.

## Full Installation

Install all optional groups at once:

```bash
pip install "arc-scope[all]"
```

This is equivalent to:

```bash
pip install "arc-scope[arc,scope,weather,optim]"
```

## From Source (Development)

Clone the repository and install in editable mode:

```bash
git clone https://github.com/MarcYin/ARCOPE.git
cd ARCOPE
pip install -e ".[dev]"
```

The `dev` group adds `pytest` and `pytest-cov` for running the test suite.

### Running Tests

```bash
python -m pytest --tb=short -q
```

### Running Tests with Coverage

```bash
python -m pytest --cov=arc_scope --cov-report=term-missing
```

## Verifying the Installation

After installation, verify that the core package loads:

```python
import arc_scope
print(arc_scope.__version__)
```

To verify optional dependencies:

```python
# Check ARC availability
try:
    from arc import arc_field
    print("ARC: available")
except ImportError:
    print("ARC: not installed (pip install 'arc-scope[arc]')")

# Check SCOPE availability
try:
    from scope import ScopeGridRunner
    print("SCOPE: available")
except ImportError:
    print("SCOPE: not installed (pip install 'arc-scope[scope]')")

# Check weather (cdsapi) availability
try:
    import cdsapi
    print("cdsapi: available")
except ImportError:
    print("cdsapi: not installed (pip install 'arc-scope[weather]')")
```
