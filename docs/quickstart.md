# Quick Start

This guide walks through your first ARC-SCOPE run using the bridge module and bundled test data. No external services or heavy dependencies are required.

## Step 1: Install the Package

```bash
pip install arc-scope
```

This installs the core package with numpy, xarray, scipy, and pandas.

## Step 2: Run the Bridge Example

The package includes bundled test data (a GeoJSON field boundary in Belgium). The bridge example creates synthetic ARC-format arrays and converts them to SCOPE-ready xarray DataArrays.

```bash
python examples/01_bridge_conversion.py
```

Or run it interactively:

```python
import numpy as np
from arc_scope.bridge import arc_arrays_to_scope_inputs
from arc_scope.bridge.parameter_map import BIO_BANDS, SCALE_BANDS

# Create synthetic ARC outputs
ny, nx, nt = 10, 10, 6
mask = np.zeros((ny, nx), dtype=bool)
mask[0, :] = True  # mask out first row

n_valid = int((~mask).sum())
rng = np.random.default_rng(42)
post_bio_tensor = rng.integers(0, 1000, size=(n_valid, 7, nt)).astype(np.float64)
scale_data = rng.random((n_valid, 15)) * 100
doys = np.array([150, 160, 170, 180, 190, 200])
geotransform = np.array([5.019, 0.001, 0.0, 51.278, 0.0, -0.001])

# Convert to SCOPE format
post_bio_da, post_bio_scale_da = arc_arrays_to_scope_inputs(
    post_bio_tensor=post_bio_tensor,
    scale_data=scale_data,
    mask=mask,
    doys=doys,
    geotransform=geotransform,
    crs="EPSG:4326",
    year=2021,
)

print("Bio DataArray:", post_bio_da.dims, post_bio_da.shape)
print("Bands:", list(post_bio_da.coords["band"].values))
```

## Step 3: Understand the Output Format

The bridge produces two xarray DataArrays:

**`post_bio_da`** -- biophysical parameters in physical units:
- Dimensions: `(y, x, band, time)`
- Bands: `N`, `cab`, `cm`, `cw`, `lai`, `ala`, `cbrown`
- Values are scaled from ARC integer codes to physical units using `BIO_SCALES`

**`post_bio_scale_da`** -- scale, phenology, and soil parameters:
- Dimensions: `(y, x, band)`
- Bands: the 7 bio scale parameters + 4 phenology + 4 soil (BSM) parameters

Both DataArrays carry spatial coordinates derived from the GDAL geotransform, and CRS metadata when rioxarray is available.

## Step 4: Next Steps

Once you are comfortable with the bridge output format, you can move to more advanced workflows:

### Full pipeline (requires ARC + SCOPE)

See [`examples/03_full_pipeline.py`](../examples/03_full_pipeline.py) for a complete configuration that runs ARC retrieval, weather fetching, and SCOPE simulation end to end.

```bash
pip install "arc-scope[all]"
python examples/03_full_pipeline.py
```

### Parameter optimisation

See [`examples/04_optimization_demo.py`](../examples/04_optimization_demo.py) for optimising SCOPE parameters against observations.

### Using local weather data

Instead of ERA5, you can supply a local CSV or NetCDF file via `LocalProvider`:

```python
from arc_scope.pipeline import PipelineConfig

config = PipelineConfig(
    geojson_path="field.geojson",
    start_date="2021-05-15",
    end_date="2021-10-01",
    crop_type="wheat",
    start_of_season=170,
    year=2021,
    weather_provider="local",
    weather_config={
        "file_path": "weather_station.csv",
        "var_map": {
            "air_temp_c": "Ta",
            "sw_down_wm2": "Rin",
            "lw_down_wm2": "Rli",
            "vapour_pressure_hpa": "ea",
            "pressure_hpa": "p",
            "wind_speed_ms": "u",
        },
    },
)
```
