# API Reference: `arc_scope.bridge`

The bridge module converts ARC retrieval outputs to SCOPE-compatible xarray DataArrays.

## Parameter Map Constants

Defined in `arc_scope.bridge.parameter_map`:

### `BIO_BANDS`

```python
BIO_BANDS: tuple[str, ...] = ("N", "cab", "cm", "cw", "lai", "ala", "cbrown")
```

SCOPE band names (lowercase) in the order stored in `post_bio_tensor`. These become the `band` coordinate of the output DataArray.

### `BIO_SCALES`

```python
BIO_SCALES: tuple[float, ...] = (
    1/100.0,    # N          (dimensionless)
    1/100.0,    # Cab        (ug cm-2)
    1/10000.0,  # Cm -> Cdm  (g cm-2)
    1/10000.0,  # Cw         (g cm-2)
    1/100.0,    # LAI        (m2 m-2)
    1/100.0,    # ala        (deg)
    1/1000.0,   # Cbrown     (dimensionless)
)
```

Scale factors applied to integer-coded ARC values to recover physical units.

### `SCALE_BANDS`

```python
SCALE_BANDS: tuple[str, ...] = (
    "N", "cab", "cm", "cw", "lai", "ala", "cbrown",  # bio (0-6)
    "n0", "m0", "n1", "m1",                           # phenology (7-10)
    "BSMBrightness", "BSMlat", "BSMlon", "SMC",       # soil (11-14)
)
```

Full 15-band names for the `scale_data` array (bio + phenology + soil).

### `ARC_BIO_INDICES`

```python
ARC_BIO_INDICES: dict[int, tuple[str, float]]
```

Mapping from `post_bio_tensor` column index to `(SCOPE_variable_name, scale_factor)`. For example, index 2 maps to `("Cdm", 1/10000.0)`.

### `ARC_SOIL_INDICES`

```python
ARC_SOIL_INDICES: dict[int, str]
```

Mapping from `scale_data` column index to SCOPE soil variable name. Indices 11-14 map to `BSMBrightness`, `BSMlat`, `BSMlon`, and `SMC`.

### `ARC_BIO_RANGES` / `ARC_SOIL_RANGES`

```python
ARC_BIO_RANGES: dict[str, tuple[float, float]]
ARC_SOIL_RANGES: dict[str, tuple[float, float]]
```

Physical value ranges for validation (from ARC's `generate_arc_refs`).

## Functions

### `arc_arrays_to_scope_inputs`

```python
def arc_arrays_to_scope_inputs(
    post_bio_tensor: np.ndarray,
    scale_data: np.ndarray,
    mask: np.ndarray,
    doys: np.ndarray,
    geotransform: np.ndarray,
    crs: Any,
    year: int,
    *,
    post_bio_unc_tensor: np.ndarray | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
```

Build SCOPE-ready xarray DataArrays from live ARC `arc_field()` outputs.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `post_bio_tensor` | `np.ndarray` | Posterior biophysical parameters, shape `(n_valid_pixels, 7, n_times)`. Integer-coded values before scaling. |
| `scale_data` | `np.ndarray` | Concatenated scale parameters, shape `(n_valid_pixels, 15)`. |
| `mask` | `np.ndarray` | Boolean mask, shape `(ny, nx)`. `True` = masked out. |
| `doys` | `np.ndarray` | 1-D array of day-of-year values, length `n_times`. |
| `geotransform` | `np.ndarray` | GDAL-style geotransform `[x_origin, x_size, x_rot, y_origin, y_rot, y_size]`. |
| `crs` | `Any` | Coordinate reference system (string, EPSG code, or WKT). |
| `year` | `int` | Calendar year to combine with `doys` for datetime coordinates. |
| `post_bio_unc_tensor` | `np.ndarray` or `None` | Optional uncertainty tensor (kept for provenance). |

**Returns:**

- `post_bio_da` -- `xr.DataArray` with dims `(y, x, band, time)` and physical units
- `post_bio_scale_da` -- `xr.DataArray` with dims `(y, x, band)` containing per-pixel scale/soil/phenology parameters

**Usage:**

```python
from arc_scope.bridge import arc_arrays_to_scope_inputs

post_bio_da, post_bio_scale_da = arc_arrays_to_scope_inputs(
    post_bio_tensor=bio_tensor,
    scale_data=scale_data,
    mask=mask,
    doys=doys,
    geotransform=geotransform,
    crs="EPSG:4326",
    year=2021,
)
```

### `arc_npz_to_scope_inputs`

```python
def arc_npz_to_scope_inputs(
    npz_path: str | Path,
    year: int,
    *,
    reference_dataset: xr.Dataset | xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
```

Load ARC outputs from a saved NPZ file and convert to SCOPE inputs. When `scope-rtm` is installed, delegates to SCOPE's `read_s2_bio_inputs`. Otherwise uses a standalone fallback.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `npz_path` | `str` or `Path` | Path to the `.npz` file saved by `arc_field()`. |
| `year` | `int` | Calendar year for datetime coordinate construction. |
| `reference_dataset` | `xr.Dataset` or `None` | Optional reference for spatial CRS alignment. |

**Returns:** Same format as `arc_arrays_to_scope_inputs`.

**Usage:**

```python
from arc_scope.bridge import arc_npz_to_scope_inputs

post_bio_da, post_bio_scale_da = arc_npz_to_scope_inputs(
    "output/arc_output.npz", year=2021
)
```

### `validate_soil_params`

```python
def validate_soil_params(
    brightness: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    smc: np.ndarray,
    *,
    strict: bool = False,
) -> dict[str, np.ndarray]:
```

Validate and optionally clip ARC BSM soil parameters for SCOPE.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `brightness` | `np.ndarray` | Soil brightness (BSMBrightness). |
| `lat` | `np.ndarray` | Soil spectral shape parameter 1 (BSMlat). |
| `lon` | `np.ndarray` | Soil spectral shape parameter 2 (BSMlon). |
| `smc` | `np.ndarray` | Soil volumetric moisture content, % (SMC). |
| `strict` | `bool` | If `True`, raise `ValueError` on out-of-range values. If `False` (default), clip to valid ranges. |

**Returns:** Dict mapping SCOPE variable names to validated arrays.

**Usage:**

```python
from arc_scope.bridge.soil import validate_soil_params

validated = validate_soil_params(
    brightness=np.array([0.3, 0.5]),
    lat=np.array([20.0, 25.0]),
    lon=np.array([40.0, 50.0]),
    smc=np.array([30.0, 60.0]),
)
```
