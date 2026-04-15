"""Tests for the weather data disk cache."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from arc_scope.weather.cache import WeatherCache


def _make_ds(n: int = 6) -> xr.Dataset:
    """Create a small weather-like dataset for caching tests."""
    times = pd.date_range("2021-06-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "Rin": ("time", rng.uniform(200, 800, n)),
            "Ta": ("time", rng.uniform(10, 30, n)),
        },
        coords={"time": times},
    )


def test_put_and_get(tmp_path):
    """Caching a dataset and retrieving it should yield equivalent data."""
    cache = WeatherCache(tmp_path / "cache")
    ds = _make_ds()
    cache.put("test_key", ds)

    loaded = cache.get("test_key")
    assert loaded is not None
    xr.testing.assert_allclose(loaded, ds)
    loaded.close()


def test_get_missing_returns_none(tmp_path):
    """Getting a key that was never cached should return None."""
    cache = WeatherCache(tmp_path / "cache")
    assert cache.get("nonexistent") is None


def test_clear_removes_files(tmp_path):
    """clear() should remove all cached NetCDF files and return the count."""
    cache = WeatherCache(tmp_path / "cache")
    ds = _make_ds()
    cache.put("a", ds)
    cache.put("b", ds)

    removed = cache.clear()
    assert removed == 2
    # After clearing, nothing should be retrievable
    assert cache.get("a") is None
    assert cache.get("b") is None


def test_cache_creates_directory(tmp_path):
    """put() should create the cache directory if it does not exist yet."""
    cache_dir = tmp_path / "new_dir" / "sub"
    cache = WeatherCache(cache_dir)

    # Directory does not exist yet
    assert not cache_dir.exists()

    cache.put("first", _make_ds())
    assert cache_dir.exists()
    assert cache.get("first") is not None
    cache.get("first").close()
