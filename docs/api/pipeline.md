# API Reference: `arc_scope.pipeline`

The pipeline module orchestrates the end-to-end workflow from field definition to SCOPE simulation.

## `PipelineConfig`

```python
@dataclass
class PipelineConfig:
    # Field definition (required)
    geojson_path: PathLike
    start_date: str
    end_date: str
    crop_type: str
    start_of_season: int
    year: int

    # ARC options
    num_samples: int = 100000
    growth_season_length: int = 45
    s2_data_folder: PathLike | None = None
    data_source: str = "aws"

    # Weather options
    weather_provider: str = "era5"
    weather_config: dict[str, Any] = field(default_factory=dict)

    # SCOPE options
    scope_workflow: str = "reflectance"
    scope_root_path: PathLike | None = None
    scope_options: dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"
    dtype: str = "float64"

    # Output options
    output_dir: PathLike = Path("./output")
    save_arc_npz: bool = True
    save_scope_netcdf: bool = True

    # Optimization
    optimize: bool = False
    optim_config: dict[str, Any] | None = None
```

Master configuration dataclass for the ARC-SCOPE pipeline.

### Field Definition (required)

| Field | Type | Description |
|-------|------|-------------|
| `geojson_path` | `PathLike` | Path to GeoJSON file defining the field boundary. |
| `start_date` | `str` | Start date for satellite data, e.g., `"2021-05-15"`. |
| `end_date` | `str` | End date for satellite data. |
| `crop_type` | `str` | Crop type identifier, e.g., `"wheat"`, `"maize"`. |
| `start_of_season` | `int` | Day of year when the growth season begins. |
| `year` | `int` | Calendar year for the simulation period. |

### ARC Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_samples` | `int` | `100000` | Number of archetype samples to generate. |
| `growth_season_length` | `int` | `45` | Length of growth season in days. |
| `s2_data_folder` | `PathLike` or `None` | `None` | Directory for caching Sentinel-2 downloads. |
| `data_source` | `str` | `"aws"` | S2 data source: `"aws"`, `"planetary"`, `"cdse"`, `"gee"`. |

### Weather Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `weather_provider` | `str` | `"era5"` | Provider name: `"era5"` or `"local"`. |
| `weather_config` | `dict` | `{}` | Provider-specific configuration passed as kwargs to the provider constructor. |

### SCOPE Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `scope_workflow` | `str` | `"reflectance"` | One of: `"reflectance"`, `"fluorescence"`, `"thermal"`, `"energy-balance"`. |
| `scope_root_path` | `PathLike` or `None` | `None` | Path to SCOPE upstream assets directory. |
| `scope_options` | `dict` | `{}` | Additional SCOPE options to override defaults. |
| `device` | `str` | `"cpu"` | PyTorch device (`"cpu"` or `"cuda"`). |
| `dtype` | `str` | `"float64"` | PyTorch dtype string. |

### Output Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_dir` | `PathLike` | `"./output"` | Directory for saving results. |
| `save_arc_npz` | `bool` | `True` | Whether to save ARC outputs to NPZ. |
| `save_scope_netcdf` | `bool` | `True` | Whether to save SCOPE outputs to NetCDF. |

### Properties

**`resolved_scope_options`** -- Merges workflow defaults with user overrides:

```python
config = PipelineConfig(..., scope_workflow="fluorescence")
config.resolved_scope_options
# {'calc_fluor': 1, 'calc_planck': 0}
```

## `ArcScopePipeline`

```python
class ArcScopePipeline:
    def __init__(self, config: PipelineConfig): ...

    def run(self) -> PipelineResult: ...
    def run_arc(self) -> ArcResult: ...
    def run_bridge(self, arc_result: ArcResult) -> tuple[xr.DataArray, xr.DataArray]: ...
    def run_weather(self) -> xr.Dataset: ...
    def run_observation(self, arc_result: ArcResult) -> xr.Dataset: ...
    def run_scope(
        self,
        post_bio_da: xr.DataArray,
        post_bio_scale_da: xr.DataArray,
        weather_ds: xr.Dataset,
        observation_ds: xr.Dataset,
    ) -> xr.Dataset: ...
```

End-to-end pipeline from field definition to SCOPE simulation.

### `run()`

Execute the full pipeline: ARC -> Bridge -> Weather -> SCOPE.

Returns a `PipelineResult` containing all intermediate and final outputs.

### `run_arc()`

Run ARC retrieval step only. Returns an `ArcResult`.

### `run_bridge(arc_result)`

Convert ARC outputs to SCOPE format. Returns `(post_bio_da, post_bio_scale_da)`.

### `run_weather()`

Fetch weather data for the configured field and time range. Returns an `xr.Dataset`.

### `run_observation(arc_result)`

Build the observation geometry dataset. Returns an `xr.Dataset`.

### `run_scope(post_bio_da, post_bio_scale_da, weather_ds, observation_ds)`

Prepare and run SCOPE from bridge outputs + weather + observations. Returns an `xr.Dataset`.

## `PipelineResult`

```python
@dataclass
class PipelineResult:
    arc_result: ArcResult | None = None
    post_bio_da: xr.DataArray | None = None
    post_bio_scale_da: xr.DataArray | None = None
    weather_ds: xr.Dataset | None = None
    observation_ds: xr.Dataset | None = None
    scope_input_ds: xr.Dataset | None = None
    scope_output_ds: xr.Dataset | None = None
```

Container for full pipeline results. All fields are `None` until their corresponding step completes.

## Step Functions

Composable building blocks for users who want partial pipelines. Defined in `arc_scope.pipeline.steps`:

### `retrieve_arc(config)`

Run ARC biophysical parameter retrieval. Requires `arc-scope[arc]`.

### `bridge_arc_to_scope(arc_result, year)`

Convert ARC retrieval outputs to SCOPE-compatible DataArrays.

### `build_observation_dataset(doys, year, geojson_path, ...)`

Build an observation geometry dataset with solar zenith/azimuth angles computed from field location and overpass time.

Optional keyword arguments: `viewing_zenith` (default 0.0), `viewing_azimuth` (default 0.0), `overpass_hour` (default 10.5 for Sentinel-2).

### `fetch_weather(config, time_range=None)`

Fetch weather data using the configured provider (`"era5"` or `"local"`).

### `prepare_scope_dataset(post_bio_da, post_bio_scale_da, weather_ds, observation_ds, config)`

Merge all inputs into a runner-ready SCOPE dataset. Requires `scope-rtm`.

### `run_scope_simulation(scope_dataset, config)`

Execute the SCOPE simulation. Requires `scope-rtm` and `torch`.

## `ArcResult`

```python
@dataclass
class ArcResult:
    scale_data: np.ndarray
    post_bio_tensor: np.ndarray
    post_bio_unc_tensor: np.ndarray
    mask: np.ndarray
    doys: np.ndarray
    geotransform: np.ndarray | None = None
    crs: Any = None
```

Container for ARC retrieval outputs.
