"""Tests for the ERA5 weather provider."""

import zipfile
from pathlib import Path

import pytest
import xarray as xr

from arc_scope.weather.era5 import (
    DEFAULT_ERA5_API_URL,
    ERA5Provider,
    _expand_bounds_for_era5,
    _open_era5_dataset,
)


def test_era5_provider_defaults_to_cds_climate_store():
    """ERA5 requests should target the climate store by default."""
    provider = ERA5Provider()
    assert provider._api_url == DEFAULT_ERA5_API_URL


def test_era5_provider_allows_api_url_override():
    """Callers may still override the CDS endpoint explicitly."""
    provider = ERA5Provider(api_url="https://example.invalid/api")
    assert provider._api_url == "https://example.invalid/api"


def test_expand_bounds_for_era5_grows_tiny_fields():
    """Very small fields should be expanded to a valid ERA5 request window."""
    bounds = (5.019294, 51.274966, 5.022639, 51.279038)
    expanded = _expand_bounds_for_era5(bounds)
    assert expanded[2] - expanded[0] == pytest.approx(0.3)
    assert expanded[3] - expanded[1] == pytest.approx(0.3)
    assert expanded[0] < bounds[0]
    assert expanded[1] < bounds[1]


def test_open_era5_dataset_reads_zipped_netcdf(tmp_path):
    """CDS ERA5 downloads may arrive as zip archives containing a NetCDF file."""
    nc_path = tmp_path / "weather.nc"
    ds = xr.Dataset({"Ta": ("time", [12.0, 13.0])}, coords={"time": [0, 1]})
    ds.to_netcdf(nc_path, engine="scipy")

    zip_path = tmp_path / "weather.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(nc_path, arcname=Path("nested") / nc_path.name)

    loaded = _open_era5_dataset(zip_path)
    assert list(loaded.data_vars) == ["Ta"]


def test_open_era5_dataset_merges_multiple_members(tmp_path):
    """ERA5 zip payloads may split instant and accumulated variables across files."""
    ta_path = tmp_path / "instant.nc"
    rin_path = tmp_path / "accum.nc"
    xr.Dataset({"Ta": ("time", [12.0, 13.0])}, coords={"time": [0, 1]}).to_netcdf(
        ta_path,
        engine="scipy",
    )
    xr.Dataset({"Rin": ("time", [100.0, 120.0])}, coords={"time": [0, 1]}).to_netcdf(
        rin_path,
        engine="scipy",
    )

    zip_path = tmp_path / "weather.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(ta_path, arcname="instant.nc")
        archive.write(rin_path, arcname="accum.nc")

    loaded = _open_era5_dataset(zip_path)
    assert sorted(loaded.data_vars) == ["Rin", "Ta"]
