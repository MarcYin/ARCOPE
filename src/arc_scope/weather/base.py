"""Abstract base class for weather data providers.

SCOPE requires the following meteorological variables (see
``scope.io.prepare.DEFAULT_WEATHER_VAR_MAP``):

- ``Rin``: Incoming shortwave radiation (W m-2)
- ``Rli``: Incoming longwave radiation (W m-2)
- ``Ta``:  Air temperature (degC)
- ``ea``:  Vapor pressure (hPa)
- ``p``:   Air pressure (hPa)
- ``u``:   Wind speed (m s-1)

For energy-balance workflows, additional variables may be needed:
- ``Ca``:  CO2 concentration (ppm)
- ``Oa``:  O2 concentration (%)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Sequence

import xarray as xr

from arc_scope.utils.types import BBox

# Variables always required by SCOPE
REQUIRED_WEATHER_VARS = ("Rin", "Rli", "Ta", "ea", "p", "u")

# Additional variables for energy-balance workflows
ENERGY_BALANCE_VARS = ("Ca", "Oa")


class WeatherProvider(ABC):
    """Abstract base class for meteorological data providers.

    Subclasses must implement :meth:`fetch` to return an
    ``xr.Dataset`` with SCOPE-compatible variable names.
    """

    @abstractmethod
    def fetch(
        self,
        bounds: BBox,
        time_range: tuple[datetime, datetime],
        variables: Sequence[str] = REQUIRED_WEATHER_VARS,
    ) -> xr.Dataset:
        """Fetch weather data for the given spatiotemporal extent.

        Parameters
        ----------
        bounds:
            Bounding box ``(minx, miny, maxx, maxy)`` in WGS84.
        time_range:
            ``(start, end)`` datetime range.
        variables:
            SCOPE variable names to fetch.

        Returns
        -------
        xr.Dataset with dims ``(time,)`` or ``(y, x, time)`` and the
        requested variables using SCOPE naming conventions.
        """
        ...

    def validate(self, ds: xr.Dataset, variables: Sequence[str] = REQUIRED_WEATHER_VARS) -> None:
        """Check that a weather dataset has all required variables."""
        missing = [v for v in variables if v not in ds]
        if missing:
            raise ValueError(f"Weather dataset missing required variables: {missing}")
