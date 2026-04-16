"""ERA5 reanalysis weather data provider via CDS API.

Maps ERA5 variable names to SCOPE conventions:
- ``2m_temperature`` -> ``Ta`` (K to degC)
- ``2m_dewpoint_temperature`` -> ``ea`` (K to hPa via Magnus formula)
- ``surface_solar_radiation_downwards`` -> ``Rin`` (J m-2 to W m-2)
- ``surface_thermal_radiation_downwards`` -> ``Rli`` (J m-2 to W m-2)
- ``surface_pressure`` -> ``p`` (Pa to hPa)
- ``10m_u_component_of_wind`` + ``10m_v_component_of_wind`` -> ``u`` (m s-1)

Requires: ``pip install cdsapi``

The ERA5 family lives in the Copernicus Climate Data Store (CDS), not the
Atmosphere Data Store (ADS). The provider therefore defaults to the CDS API
base URL even when a user's ``~/.cdsapirc`` is configured for ADS.
"""

from __future__ import annotations

import hashlib
import tempfile
import zipfile
from calendar import monthrange
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr

from arc_scope.utils.types import BBox
from arc_scope.weather.base import REQUIRED_WEATHER_VARS, WeatherProvider
from arc_scope.weather.cache import WeatherCache

# ERA5 variable names to request
ERA5_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
DEFAULT_ERA5_API_URL = "https://cds.climate.copernicus.eu/api"
ERA5_MIN_AREA_SPAN_DEGREES = 0.3
ERA5_CACHE_VERSION = "v2"


class ERA5Provider(WeatherProvider):
    """Fetch ERA5 hourly reanalysis data from the Copernicus Climate Data Store.

    Parameters
    ----------
    cache_dir:
        Directory for caching downloaded files. Defaults to
        ``~/.cache/arc_scope/weather/era5``.
    product:
        CDS product name. Default: ``"reanalysis-era5-single-levels"``.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        product: str = "reanalysis-era5-single-levels",
        api_url: str = DEFAULT_ERA5_API_URL,
    ):
        self._product = product
        self._api_url = api_url
        self._cache = WeatherCache(
            cache_dir or Path.home() / ".cache" / "arc_scope" / "weather" / "era5"
        )

    def fetch(
        self,
        bounds: BBox,
        time_range: tuple[datetime, datetime],
        variables: Sequence[str] = REQUIRED_WEATHER_VARS,
    ) -> xr.Dataset:
        """Fetch ERA5 data, using cache when available."""
        cache_key = self._cache_key(bounds, time_range)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        ds = self._download(bounds, time_range)
        ds = self._convert_to_scope(ds)
        self._cache.put(cache_key, ds)
        return ds

    def _download(self, bounds: BBox, time_range: tuple[datetime, datetime]) -> xr.Dataset:
        """Download ERA5 data via CDS API."""
        try:
            import cdsapi
        except ImportError:
            raise ImportError(
                "cdsapi is required for ERA5 downloads. "
                "Install with: pip install arc-scope[weather]"
            )

        minx, miny, maxx, maxy = _expand_bounds_for_era5(bounds)
        start, end = time_range

        client = cdsapi.Client(url=self._api_url)
        datasets: list[xr.Dataset] = []
        for chunk_start, chunk_end in _iter_month_windows(start, end):
            dates = _iter_dates(chunk_start, chunk_end)
            with tempfile.NamedTemporaryFile(suffix=".download", delete=False) as tmp:
                tmp_path = tmp.name

            client.retrieve(
                self._product,
                {
                    "product_type": "reanalysis",
                    "variable": ERA5_VARIABLES,
                    "year": sorted({d[:4] for d in dates}),
                    "month": sorted({d[5:7] for d in dates}),
                    "day": sorted({d[8:10] for d in dates}),
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "area": [maxy, minx, miny, maxx],  # CDS uses [N, W, S, E]
                    "data_format": "netcdf",
                },
                tmp_path,
            )

            ds = _open_era5_dataset(Path(tmp_path))
            ds.load()
            Path(tmp_path).unlink(missing_ok=True)
            datasets.append(ds)

        if len(datasets) == 1:
            return datasets[0]

        time_dim = "valid_time" if "valid_time" in datasets[0].dims else "time"
        merged = xr.concat(datasets, dim=time_dim).sortby(time_dim)
        merged = merged.isel({time_dim: ~merged.get_index(time_dim).duplicated()})
        return merged

    def _convert_to_scope(self, ds: xr.Dataset) -> xr.Dataset:
        """Convert ERA5 variables to SCOPE naming and units."""
        if "valid_time" in ds.dims or "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})

        out = xr.Dataset()

        # Air temperature: K -> degC
        if "t2m" in ds:
            out["Ta"] = ds["t2m"] - 273.15
        elif "2m_temperature" in ds:
            out["Ta"] = ds["2m_temperature"] - 273.15

        # Vapor pressure from dewpoint temperature: K -> hPa
        if "d2m" in ds:
            td = ds["d2m"] - 273.15
        elif "2m_dewpoint_temperature" in ds:
            td = ds["2m_dewpoint_temperature"] - 273.15
        else:
            td = None
        if td is not None:
            # Magnus formula: ea = 6.108 * exp(17.27 * Td / (Td + 237.3))
            out["ea"] = 6.108 * np.exp(17.27 * td / (td + 237.3))

        # Shortwave radiation: J m-2 (accumulated) -> W m-2 (hourly mean)
        for era5_name in ("ssrd", "surface_solar_radiation_downwards"):
            if era5_name in ds:
                out["Rin"] = ds[era5_name] / 3600.0
                break

        # Longwave radiation: J m-2 (accumulated) -> W m-2 (hourly mean)
        for era5_name in ("strd", "surface_thermal_radiation_downwards"):
            if era5_name in ds:
                out["Rli"] = ds[era5_name] / 3600.0
                break

        # Pressure: Pa -> hPa
        for era5_name in ("sp", "surface_pressure"):
            if era5_name in ds:
                out["p"] = ds[era5_name] / 100.0
                break

        # Wind speed: magnitude from u and v components
        u_var = next((v for v in ("u10", "10m_u_component_of_wind") if v in ds), None)
        v_var = next((v for v in ("v10", "10m_v_component_of_wind") if v in ds), None)
        if u_var and v_var:
            out["u"] = np.sqrt(ds[u_var] ** 2 + ds[v_var] ** 2)

        # Copy time coordinate
        if "time" in ds.coords:
            out = out.assign_coords(time=ds.coords["time"])

        # Spatially average to a single point for field-scale use
        for spatial_dim in ("latitude", "longitude", "lat", "lon"):
            if spatial_dim in out.dims:
                out = out.mean(dim=spatial_dim)

        return out

    def _cache_key(self, bounds: BBox, time_range: tuple[datetime, datetime]) -> str:
        """Generate a deterministic cache key."""
        raw = (
            f"{ERA5_CACHE_VERSION}|{self._product}|{self._api_url}|"
            f"{bounds}_{time_range[0].isoformat()}_{time_range[1].isoformat()}"
        )
        return hashlib.md5(raw.encode()).hexdigest()


def _iter_dates(start: datetime, end: datetime) -> list[str]:
    """Return inclusive daily ISO dates for a datetime range."""
    dates: list[str] = []
    current = start.replace(hour=0, minute=0, second=0, microsecond=0)
    final = end.replace(hour=0, minute=0, second=0, microsecond=0)
    while current <= final:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def _iter_month_windows(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Split a long ERA5 request into month-sized chunks to stay under CDS limits."""
    windows: list[tuple[datetime, datetime]] = []
    current = start
    while current <= end:
        last_day = monthrange(current.year, current.month)[1]
        month_end = current.replace(
            day=last_day,
            hour=23,
            minute=59,
            second=59,
            microsecond=999999,
        )
        window_end = min(end, month_end)
        windows.append((current, window_end))
        if window_end >= end:
            break
        next_day = (window_end + timedelta(days=1)).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        current = next_day
    return windows


def _expand_bounds_for_era5(bounds: BBox) -> BBox:
    """Expand tiny field bounds so a gridded ERA5 request hits at least one cell."""
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny
    if width >= ERA5_MIN_AREA_SPAN_DEGREES and height >= ERA5_MIN_AREA_SPAN_DEGREES:
        return bounds

    half_span = ERA5_MIN_AREA_SPAN_DEGREES / 2.0
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    expanded = (
        max(-180.0, cx - half_span),
        max(-90.0, cy - half_span),
        min(180.0, cx + half_span),
        min(90.0, cy + half_span),
    )
    return expanded


def _open_era5_dataset(path: Path) -> xr.Dataset:
    """Open a CDS ERA5 payload, handling zipped NetCDF downloads when needed."""
    if zipfile.is_zipfile(path):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(path) as archive:
                members = [name for name in archive.namelist() if name.endswith(".nc")]
                if not members:
                    raise ValueError(f"ERA5 archive {path} did not contain a NetCDF payload.")
                extracted_paths = [Path(archive.extract(name, path=tmp_dir)) for name in members]

            datasets: list[xr.Dataset] = []
            for extracted in extracted_paths:
                ds = _open_netcdf_with_available_engine(extracted)
                ds.load()
                datasets.append(ds)
            if len(datasets) == 1:
                return datasets[0]
            return xr.merge(datasets, compat="override", combine_attrs="override")

    return _open_netcdf_with_available_engine(path)


def _open_netcdf_with_available_engine(path: Path) -> xr.Dataset:
    """Open NetCDF using the first available xarray backend that supports the file."""
    last_error: Exception | None = None
    for engine in ("netcdf4", "h5netcdf", "scipy"):
        try:
            return xr.open_dataset(path, engine=engine)
        except Exception as exc:  # pragma: no cover - exercised by runtime backends
            last_error = exc
    raise ValueError(f"Could not open ERA5 NetCDF payload at {path}") from last_error
