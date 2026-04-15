"""Tests for the weather provider base class and constants."""
from __future__ import annotations

import pytest
import xarray as xr

from arc_scope.weather.base import (
    ENERGY_BALANCE_VARS,
    REQUIRED_WEATHER_VARS,
    WeatherProvider,
)


def test_validate_passes_with_all_vars(sample_weather_ds):
    """validate() should succeed when the dataset contains all required variables."""
    # WeatherProvider is abstract, but validate() is concrete -- we can call it
    # on any instance.  Create a minimal concrete subclass.
    class _Dummy(WeatherProvider):
        def fetch(self, bounds, time_range, variables=REQUIRED_WEATHER_VARS):
            return sample_weather_ds

    provider = _Dummy()
    # Should not raise
    provider.validate(sample_weather_ds)


def test_validate_raises_on_missing():
    """validate() should raise ValueError when a required variable is absent."""
    ds = xr.Dataset({"Rin": ("time", [1.0]), "Rli": ("time", [2.0])})

    class _Dummy(WeatherProvider):
        def fetch(self, bounds, time_range, variables=REQUIRED_WEATHER_VARS):
            return ds

    provider = _Dummy()
    with pytest.raises(ValueError, match="missing required variables"):
        provider.validate(ds)


def test_required_vars_tuple_length():
    """REQUIRED_WEATHER_VARS should contain exactly 6 variable names."""
    assert len(REQUIRED_WEATHER_VARS) == 6
    for v in ("Rin", "Rli", "Ta", "ea", "p", "u"):
        assert v in REQUIRED_WEATHER_VARS


def test_energy_balance_vars():
    """ENERGY_BALANCE_VARS should contain Ca and Oa."""
    assert len(ENERGY_BALANCE_VARS) == 2
    assert "Ca" in ENERGY_BALANCE_VARS
    assert "Oa" in ENERGY_BALANCE_VARS
