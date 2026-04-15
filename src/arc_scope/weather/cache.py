"""Simple disk cache for weather data downloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from arc_scope.utils.types import PathLike


def _netcdf_encoding(ds: xr.Dataset) -> dict:
    """Build CF-compliant encoding so scipy can write datetime coords."""
    encoding: dict = {}
    for name, var in ds.coords.items():
        if np.issubdtype(var.dtype, np.datetime64):
            encoding[name] = {"units": "hours since 2000-01-01", "dtype": "float64"}
    for name, var in ds.data_vars.items():
        if np.issubdtype(var.dtype, np.datetime64):
            encoding[name] = {"units": "hours since 2000-01-01", "dtype": "float64"}
    return encoding


class WeatherCache:
    """File-system cache for xarray weather datasets.

    Stores datasets as NetCDF files keyed by a string identifier.

    Parameters
    ----------
    cache_dir:
        Directory for cached files.  Created on first use.
    """

    def __init__(self, cache_dir: PathLike):
        self._dir = Path(cache_dir)

    def get(self, key: str) -> xr.Dataset | None:
        """Retrieve a cached dataset, or ``None`` if not found."""
        path = self._path(key)
        if path.exists():
            return xr.open_dataset(path, engine="scipy")
        return None

    def put(self, key: str, ds: xr.Dataset) -> Path:
        """Cache a dataset and return the file path."""
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        ds.to_netcdf(path, engine="scipy", encoding=_netcdf_encoding(ds))
        return path

    def clear(self) -> int:
        """Remove all cached files. Returns the number removed."""
        if not self._dir.exists():
            return 0
        count = 0
        for f in self._dir.glob("*.nc"):
            f.unlink()
            count += 1
        return count

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.nc"
