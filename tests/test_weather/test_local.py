"""Tests for the local weather data provider."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arc_scope.weather.local import LocalProvider


# ---------------------------------------------------------------------------
# Helper to build a simple CSV weather file
# ---------------------------------------------------------------------------

def _write_csv(path, n=24):
    """Write a minimal weather CSV with ``n`` hourly rows."""
    times = pd.date_range("2021-06-01", periods=n, freq="h")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "time": times,
        "air_temp_c": rng.uniform(10, 30, n),
        "sw_down": rng.uniform(100, 800, n),
        "lw_down": rng.uniform(250, 400, n),
        "vapour_press": rng.uniform(5, 25, n),
        "pressure": rng.uniform(980, 1030, n),
        "wind": rng.uniform(0.5, 10, n),
    })
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_csv_weather(tmp_path):
    """Load a temporary CSV and verify the resulting dataset has SCOPE names."""
    csv_path = tmp_path / "weather.csv"
    _write_csv(csv_path)

    var_map = {
        "air_temp_c": "Ta",
        "sw_down": "Rin",
        "lw_down": "Rli",
        "vapour_press": "ea",
        "pressure": "p",
        "wind": "u",
    }
    provider = LocalProvider(csv_path, var_map=var_map)
    ds = provider.fetch(
        bounds=(5.0, 51.0, 5.1, 51.1),
        time_range=(datetime(2021, 6, 1), datetime(2021, 6, 2)),
    )
    for v in ("Ta", "Rin", "Rli", "ea", "p", "u"):
        assert v in ds, f"Expected variable {v} in dataset"
    assert "time" in ds.dims


def test_load_netcdf_weather(tmp_path):
    """Load a temporary NetCDF and verify correct variable mapping."""
    nc_path = tmp_path / "weather.nc"
    times = pd.date_range("2021-06-01", periods=12, freq="h")
    rng = np.random.default_rng(7)
    ds_orig = xr.Dataset(
        {
            "temperature": ("time", rng.uniform(10, 30, 12)),
            "shortwave": ("time", rng.uniform(100, 800, 12)),
        },
        coords={"time": times},
    )
    ds_orig.to_netcdf(nc_path)

    var_map = {"temperature": "Ta", "shortwave": "Rin"}
    provider = LocalProvider(nc_path, var_map=var_map)
    ds = provider.fetch(
        bounds=(5.0, 51.0, 5.1, 51.1),
        time_range=(datetime(2021, 6, 1), datetime(2021, 6, 1, 12)),
    )
    assert "Ta" in ds
    assert "Rin" in ds


def test_var_map_rename(tmp_path):
    """Only columns present in var_map should be renamed; others stay as-is."""
    csv_path = tmp_path / "weather.csv"
    _write_csv(csv_path)

    # Map only one variable
    var_map = {"air_temp_c": "Ta"}
    provider = LocalProvider(csv_path, var_map=var_map)
    ds = provider.fetch(
        bounds=(5.0, 51.0, 5.1, 51.1),
        time_range=(datetime(2021, 6, 1), datetime(2021, 6, 2)),
    )
    assert "Ta" in ds
    # The original un-mapped column should still exist under its CSV name
    assert "sw_down" in ds


def test_time_range_filtering(tmp_path):
    """Only rows within the requested time range should be returned."""
    csv_path = tmp_path / "weather.csv"
    _write_csv(csv_path, n=48)  # 2 full days

    var_map = {"air_temp_c": "Ta"}
    provider = LocalProvider(csv_path, var_map=var_map)
    ds = provider.fetch(
        bounds=(5.0, 51.0, 5.1, 51.1),
        time_range=(datetime(2021, 6, 1, 6), datetime(2021, 6, 1, 18)),
    )
    # Should have fewer than 48 timesteps
    assert ds.sizes["time"] < 48
    assert ds.sizes["time"] > 0


def test_missing_file_raises():
    """Constructing a LocalProvider with a non-existent path should raise."""
    with pytest.raises(FileNotFoundError, match="Weather file not found"):
        LocalProvider("/does/not/exist.csv", var_map={})
