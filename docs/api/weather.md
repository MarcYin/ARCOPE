# API Reference: `arc_scope.weather`

The weather module provides meteorological forcing data for SCOPE simulations through a pluggable provider interface.

## Required Weather Variables

SCOPE requires six meteorological variables:

| Variable | Name | Units |
|----------|------|-------|
| `Rin` | Incoming shortwave radiation | W m-2 |
| `Rli` | Incoming longwave radiation | W m-2 |
| `Ta` | Air temperature | degC |
| `ea` | Vapor pressure | hPa |
| `p` | Air pressure | hPa |
| `u` | Wind speed | m s-1 |

For energy-balance workflows, additional variables may be needed:

| Variable | Name | Units |
|----------|------|-------|
| `Ca` | CO2 concentration | ppm |
| `Oa` | O2 concentration | % |

## `WeatherProvider` (ABC)

```python
class WeatherProvider(ABC):
    @abstractmethod
    def fetch(
        self,
        bounds: BBox,
        time_range: tuple[datetime, datetime],
        variables: Sequence[str] = REQUIRED_WEATHER_VARS,
    ) -> xr.Dataset: ...

    def validate(
        self,
        ds: xr.Dataset,
        variables: Sequence[str] = REQUIRED_WEATHER_VARS,
    ) -> None: ...
```

Abstract base class for meteorological data providers.

**`fetch()`** retrieves weather data for the given spatiotemporal extent. Returns an `xr.Dataset` with dims `(time,)` or `(y, x, time)` using SCOPE naming conventions.

**`validate()`** checks that a weather dataset has all required variables, raising `ValueError` if any are missing.

### Parameters for `fetch()`

| Name | Type | Description |
|------|------|-------------|
| `bounds` | `BBox` | Bounding box `(minx, miny, maxx, maxy)` in WGS84. |
| `time_range` | `tuple[datetime, datetime]` | Start and end datetime. |
| `variables` | `Sequence[str]` | SCOPE variable names to fetch. Defaults to `REQUIRED_WEATHER_VARS`. |

## `ERA5Provider`

```python
class ERA5Provider(WeatherProvider):
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        product: str = "reanalysis-era5-single-levels",
    ): ...
```

Fetch ERA5 hourly reanalysis data from the Copernicus Climate Data Store.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `cache_dir` | `str`, `Path`, or `None` | `~/.cache/arc_scope/weather/era5` | Directory for caching downloaded files. |
| `product` | `str` | `"reanalysis-era5-single-levels"` | CDS product name. |

**Requirements:** `pip install arc-scope[weather]` and a valid `~/.cdsapirc` file.

**Unit conversions performed:**

- Temperature: K to degC
- Dewpoint to vapor pressure: Magnus formula
- Radiation: J m-2 (accumulated) to W m-2 (hourly mean, divided by 3600)
- Pressure: Pa to hPa
- Wind: u/v components combined to scalar magnitude

**Usage:**

```python
from datetime import datetime
from arc_scope.weather.era5 import ERA5Provider

provider = ERA5Provider()
ds = provider.fetch(
    bounds=(5.019, 51.275, 5.023, 51.279),
    time_range=(datetime(2021, 6, 1), datetime(2021, 9, 30)),
)
```

## `LocalProvider`

```python
class LocalProvider(WeatherProvider):
    def __init__(
        self,
        file_path: PathLike,
        var_map: Mapping[str, str],
        *,
        time_column: str = "time",
        time_format: str | None = None,
    ): ...
```

Load weather data from a local CSV or NetCDF file.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `file_path` | `PathLike` | *required* | Path to `.csv`, `.nc`, or `.nc4` file. |
| `var_map` | `Mapping[str, str]` | *required* | Mapping from file column/variable names to SCOPE names. |
| `time_column` | `str` | `"time"` | Column name for timestamps in CSV files. |
| `time_format` | `str` or `None` | `None` | `strptime` format for time parsing. Auto-inferred if `None`. |

**Usage:**

```python
from datetime import datetime
from arc_scope.weather.local import LocalProvider

provider = LocalProvider(
    file_path="weather_data.csv",
    var_map={
        "air_temp_c": "Ta",
        "sw_radiation": "Rin",
        "lw_radiation": "Rli",
        "vapour_pressure": "ea",
        "pressure": "p",
        "wind_speed": "u",
    },
    time_column="timestamp",
)
ds = provider.fetch(
    bounds=(5.0, 51.0, 6.0, 52.0),
    time_range=(datetime(2021, 6, 1), datetime(2021, 9, 30)),
)
```

## Radiation Partitioning

Defined in `arc_scope.weather.radiation`:

### `partition_shortwave`

```python
def partition_shortwave(
    rin: np.ndarray,
    sza: np.ndarray,
    doy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
```

Partition total incoming shortwave radiation into direct (beam) and diffuse components using the BRL diffuse-fraction model.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `rin` | `np.ndarray` | Total incoming shortwave radiation (W m-2). |
| `sza` | `np.ndarray` | Solar zenith angle (degrees). |
| `doy` | `np.ndarray` | Day of year. |

**Returns:** `(direct, diffuse)` tuple of arrays in W m-2.

### `diffuse_fraction_brl`

```python
def diffuse_fraction_brl(kt: np.ndarray) -> np.ndarray:
```

Estimate the diffuse fraction using the Boland-Ridley-Lauret logistic model.

### `extraterrestrial_irradiance`

```python
def extraterrestrial_irradiance(doy: np.ndarray) -> np.ndarray:
```

Compute top-of-atmosphere solar irradiance accounting for Earth-Sun distance using the Spencer (1971) correction.

## `WeatherCache`

```python
class WeatherCache:
    def __init__(self, cache_dir: PathLike): ...
    def get(self, key: str) -> xr.Dataset | None: ...
    def put(self, key: str, ds: xr.Dataset) -> Path: ...
    def clear(self) -> int: ...
```

Simple file-system cache for xarray weather datasets, stored as NetCDF files keyed by a string identifier. Used internally by `ERA5Provider`.
