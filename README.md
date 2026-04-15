# ARC-SCOPE

**Bridge ARC crop-parameter retrieval with SCOPE radiative-transfer simulations.**

[![Tests](https://github.com/MarcYin/ARCOPE/actions/workflows/tests.yml/badge.svg)](https://github.com/MarcYin/ARCOPE/actions/workflows/tests.yml)
![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.11%20%7C%203.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

ARC-SCOPE is an integration package that connects the **ARC** (Automated Retrieval of Crop biophysical parameters) system with the **SCOPE** (Soil-Canopy Observation of Photosynthesis and Energy fluxes) radiative-transfer model.  It converts Sentinel-2-derived biophysical parameters into SCOPE-ready inputs, fetches meteorological forcing data, orchestrates the full simulation pipeline, and supports parameter optimisation against observed SIF, thermal, or flux data.

The strongest verified path in this repository today is the **core-dependency showcase experiment**. Full SCOPE execution is an optional downstream integration rather than the default first run.

## Architecture

```
                         ARC-SCOPE Data Flow
 ================================================================

  GeoJSON + Dates ──> ARC Retrieval
                         |
                         v
                    post_bio_tensor ──> Bridge ──> post_bio_da
                    scale_data              |      post_bio_scale_da
                                            |
                    ERA5 / Local ───> Weather Provider ──> weather_ds
                                            |
                    Field centroid ──> Observation Geometry ──> obs_ds
                                            |
                                            v
                                   prepare_scope_dataset()
                                            |
                                            v
                                     SCOPE Simulation
                                     (PyTorch runner)
                                            |
                                            v
                                  Reflectance / SIF / LST
                                  Energy-balance fluxes
                                            |
                                            v
                                   [Optional] Optimisation
                                   Tune fqe, rss, rbs, ...
                                   against observations
```

## Installation

**Core package** (numpy, xarray, scipy, pandas only):

```bash
pip install arc-scope
```

**With ARC satellite retrieval** (requires GDAL):

```bash
pip install "arc-scope[arc]"
```

**With SCOPE simulation** (PyTorch):

```bash
pip install "arc-scope[scope]"
```

**With ERA5 weather downloads**:

```bash
pip install "arc-scope[weather]"
```

**Everything**:

```bash
pip install "arc-scope[all]"
```

**Development**:

```bash
pip install -e ".[dev]"
```

See [docs/installation.md](docs/installation.md) for detailed installation instructions including GDAL setup and ERA5 credential configuration.

## Quick Start

### 1. Showcase experiment (core dependencies only)

Run the primary in-repo showcase to assemble SCOPE-shaped inputs, inspect forcing diagnostics, and fit a proxy fluorescence response without requiring ARC, SCOPE, or ERA5 credentials:

```bash
pip install arc-scope
python3 -m arc_scope.experiments.showcase --output-dir ./showcase-output
```

If you are working from a repo checkout, `examples/05_showcase_experiment.py` wraps the same packaged entry point.

See [docs/showcase-experiment.md](docs/showcase-experiment.md) for the full walkthrough and generated artifacts.

### 2. Bridge conversion (standalone, from NPZ)

Convert saved ARC outputs to SCOPE-ready xarray DataArrays without needing ARC or SCOPE installed:

```python
from arc_scope.bridge import arc_npz_to_scope_inputs

post_bio_da, post_bio_scale_da = arc_npz_to_scope_inputs(
    "arc_output.npz", year=2021
)
print(post_bio_da.dims)   # ('y', 'x', 'band', 'time')
print(post_bio_da.coords["band"].values)  # ['N', 'cab', 'cm', ...]
```

### 3. Full pipeline (requires ARC + SCOPE)

Run the complete workflow from a GeoJSON field boundary to SCOPE outputs:

```python
from arc_scope.pipeline import ArcScopePipeline, PipelineConfig

config = PipelineConfig(
    geojson_path="field.geojson",
    start_date="2021-05-15",
    end_date="2021-10-01",
    crop_type="wheat",
    start_of_season=170,
    year=2021,
    scope_workflow="fluorescence",
)
pipeline = ArcScopePipeline(config)
result = pipeline.run()
```

### 4. Parameter optimisation

Optimise SCOPE parameters (e.g., fluorescence quantum efficiency) against observations:

```python
from arc_scope.optim.parameters import ParameterSet, ParameterSpec
from arc_scope.optim.protocols import ScipyOptimizer
from arc_scope.optim.objective import ScopeObjective

params = ParameterSet([
    ParameterSpec("fqe", initial=0.01, lower=0.001, upper=0.1, transform="log"),
])
optimizer = ScipyOptimizer(method="L-BFGS-B", max_iter=50)
optimised = optimizer.step(objective, params)
```

## Module Overview

| Module | Purpose | Key classes/functions |
|--------|---------|----------------------|
| `arc_scope.bridge` | Convert ARC arrays to SCOPE format | `arc_arrays_to_scope_inputs`, `arc_npz_to_scope_inputs`, `validate_soil_params` |
| `arc_scope.weather` | Fetch meteorological forcing data | `WeatherProvider`, `ERA5Provider`, `LocalProvider` |
| `arc_scope.pipeline` | End-to-end orchestration | `PipelineConfig`, `ArcScopePipeline` |
| `arc_scope.optim` | Parameter optimisation | `ParameterSpec`, `ParameterSet`, `ScopeObjective`, `ScipyOptimizer`, `TorchOptimizer` |
| `arc_scope.experiments` | Reproducible showcase experiments | `run_showcase_experiment`, `write_showcase_artifacts` |
| `arc_scope.utils` | Geometry, I/O, type aliases | `solar_position`, `load_geojson_bounds`, `BBox`, `PathLike` |
| `arc_scope.data` | Bundled test data | `TEST_FIELD_GEOJSON`, `SHOWCASE_WEATHER_CSV` |

## Configuration Reference

`PipelineConfig` is a dataclass that controls the entire pipeline:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `geojson_path` | `PathLike` | *required* | Path to field boundary GeoJSON |
| `start_date` | `str` | *required* | Start date (`YYYY-MM-DD`) |
| `end_date` | `str` | *required* | End date (`YYYY-MM-DD`) |
| `crop_type` | `str` | *required* | Crop identifier (e.g., `"wheat"`) |
| `start_of_season` | `int` | *required* | Growth season start (day of year) |
| `year` | `int` | *required* | Calendar year |
| `num_samples` | `int` | `100000` | ARC archetype samples |
| `growth_season_length` | `int` | `45` | Season length in days |
| `weather_provider` | `str` | `"era5"` | `"era5"` or `"local"` |
| `scope_workflow` | `str` | `"reflectance"` | Simulation workflow |
| `device` | `str` | `"cpu"` | PyTorch device |
| `output_dir` | `PathLike` | `"./output"` | Output directory |

## SCOPE Workflows

| Workflow | `calc_fluor` | `calc_planck` | Description |
|----------|:---:|:---:|-------------|
| `reflectance` | 0 | 0 | Directional reflectance only (fastest) |
| `fluorescence` | 1 | 0 | Reflectance + SIF (F685, F740) |
| `thermal` | 0 | 1 | Reflectance + thermal emission (LST) |
| `energy-balance` | 1 | 1 | Full energy balance with SIF + thermal + fluxes |

## Examples

Working examples are in the [`examples/`](examples/) directory:

- **[01_bridge_conversion.py](examples/01_bridge_conversion.py)** -- Convert synthetic ARC arrays to SCOPE format (no external dependencies)
- **[02_reflectance_simulation.py](examples/02_reflectance_simulation.py)** -- Prepare and inspect a SCOPE-ready dataset
- **[03_full_pipeline.py](examples/03_full_pipeline.py)** -- Complete pipeline configuration and step-by-step execution
- **[04_optimization_demo.py](examples/04_optimization_demo.py)** -- Parameter optimisation workflow with transforms and injection
- **[05_showcase_experiment.py](examples/05_showcase_experiment.py)** -- Core-only showcase experiment with local weather, geometry, radiation partitioning, and proxy calibration

## Development

```bash
git clone https://github.com/MarcYin/ARCOPE.git
cd ARCOPE
pip install -e ".[dev]"
python -m pytest --tb=short -q
```

## License

MIT -- see [LICENSE](LICENSE) for details.
