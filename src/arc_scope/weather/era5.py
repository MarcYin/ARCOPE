"""ERA5 reanalysis weather data provider via CDS API.

Maps ERA5 variable names to SCOPE conventions:
- ``2m_temperature`` -> ``Ta`` (K to degC)
- ``2m_dewpoint_temperature`` -> ``ea`` (K to hPa via Magnus formula)
- ``surface_solar_radiation_downwards`` -> ``Rin`` (J m-2 to W m-2)
- ``surface_thermal_radiation_downwards`` -> ``Rli`` (J m-2 to W m-2)
- ``surface_pressure`` -> ``p`` (Pa to hPa)
- ``10m_u_component_of_wind`` + ``10m_v_component_of_wind`` -> ``u`` (m s-1)

Requires: ``pip install cdsapi``
"""

from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime
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
    ):
        self._product = product
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

        minx, miny, maxx, maxy = bounds
        start, end = time_range

        # Build date and time lists
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current = current.replace(day=current.day + 1) if current.day < 28 else current.replace(month=current.month + 1, day=1)

        # Unique dates
        dates = sorted(set(dates))

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = tmp.name

        client = cdsapi.Client()
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
                "format": "netcdf",
            },
            tmp_path,
        )

        ds = xr.open_dataset(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
        return ds

    def _convert_to_scope(self, ds: xr.Dataset) -> xr.Dataset:
        """Convert ERA5 variables to SCOPE naming and units."""
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
        raw = f"{bounds}_{time_range[0].isoformat()}_{time_range[1].isoformat()}"
        return hashlib.md5(raw.encode()).hexdigest()
