"""Local weather data provider for station CSV or NetCDF files.

Users supply a file and a column/variable mapping to SCOPE variable names.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import xarray as xr

from arc_scope.utils.types import BBox, PathLike
from arc_scope.weather.base import REQUIRED_WEATHER_VARS, WeatherProvider


class LocalProvider(WeatherProvider):
    """Load weather data from a local CSV or NetCDF file.

    Parameters
    ----------
    file_path:
        Path to the weather data file (``.csv``, ``.nc``, ``.nc4``).
    var_map:
        Mapping from file column/variable names to SCOPE names.
        Example: ``{"air_temp_c": "Ta", "sw_down": "Rin", ...}``
    time_column:
        Column name for timestamps in CSV files. Ignored for NetCDF.
    time_format:
        ``strptime`` format for parsing time strings in CSV. If ``None``,
        ``pd.to_datetime`` infers the format automatically.
    """

    def __init__(
        self,
        file_path: PathLike,
        var_map: Mapping[str, str],
        *,
        time_column: str = "time",
        time_format: str | None = None,
    ):
        self._path = Path(file_path)
        self._var_map = dict(var_map)
        self._time_column = time_column
        self._time_format = time_format

        if not self._path.exists():
            raise FileNotFoundError(f"Weather file not found: {self._path}")

    def fetch(
        self,
        bounds: BBox,
        time_range: tuple[datetime, datetime],
        variables: Sequence[str] = REQUIRED_WEATHER_VARS,
    ) -> xr.Dataset:
        """Load and rename weather data from the local file."""
        suffix = self._path.suffix.lower()

        if suffix == ".csv":
            ds = self._load_csv(time_range)
        elif suffix in (".nc", ".nc4", ".netcdf"):
            ds = self._load_netcdf(time_range)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        return ds

    def _load_csv(self, time_range: tuple[datetime, datetime]) -> xr.Dataset:
        """Load from CSV with time parsing and variable renaming."""
        df = pd.read_csv(self._path)

        if self._time_column in df.columns:
            df[self._time_column] = pd.to_datetime(
                df[self._time_column], format=self._time_format
            )
            df = df.set_index(self._time_column)
            df.index.name = "time"

        # Filter time range
        start, end = time_range
        df = df.loc[str(start):str(end)]

        # Rename columns
        rename = {src: dst for src, dst in self._var_map.items() if src in df.columns}
        df = df.rename(columns=rename)

        return xr.Dataset.from_dataframe(df)

    def _load_netcdf(self, time_range: tuple[datetime, datetime]) -> xr.Dataset:
        """Load from NetCDF with variable renaming."""
        ds = xr.open_dataset(self._path, engine="scipy")

        # Time slicing
        if "time" in ds.dims:
            start, end = time_range
            ds = ds.sel(time=slice(str(start), str(end)))

        # Rename variables
        rename = {src: dst for src, dst in self._var_map.items() if src in ds}
        ds = ds.rename(rename)

        return ds
