"""Tests for individual pipeline step functions."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import xarray as xr

from arc_scope.data import TEST_FIELD_GEOJSON
from arc_scope.pipeline.steps import (
    ArcResult,
    bridge_arc_to_scope,
    build_observation_dataset,
    fetch_weather,
    retrieve_arc,
)
from arc_scope.pipeline.config import PipelineConfig


# ---------------------------------------------------------------------------
# build_observation_dataset tests
# ---------------------------------------------------------------------------

_DOYS = np.array([150, 160, 170, 180, 190, 200])


def test_build_observation_dataset_shapes():
    """Observation dataset should have the right variables and matching sizes."""
    ds = build_observation_dataset(
        doys=_DOYS, year=2021, geojson_path=TEST_FIELD_GEOJSON,
    )
    assert "solar_zenith_angle" in ds
    assert "viewing_zenith_angle" in ds
    assert "solar_azimuth_angle" in ds
    assert "viewing_azimuth_angle" in ds
    assert "delta_time" in ds
    assert ds.sizes["time"] == len(_DOYS)


def test_build_observation_dataset_solar_angles_reasonable():
    """Solar zenith angles for summer mid-latitudes should be < 90 degrees."""
    ds = build_observation_dataset(
        doys=_DOYS, year=2021, geojson_path=TEST_FIELD_GEOJSON,
    )
    sza = ds["solar_zenith_angle"].values
    # All summer observations in the Netherlands at ~10:30 should be < 70 deg
    assert np.all(sza > 0), "SZA should be positive"
    assert np.all(sza < 90), "SZA should be < 90 during daytime"


def test_build_observation_dataset_time_coords():
    """Time coordinates should be datetime64 values in the correct year."""
    ds = build_observation_dataset(
        doys=_DOYS, year=2021, geojson_path=TEST_FIELD_GEOJSON,
    )
    times = ds.coords["time"].values
    assert len(times) == len(_DOYS)
    for t in times:
        year_str = str(t)[:4]
        assert year_str == "2021"


def test_build_observation_dataset_offsets_duplicate_days():
    """Repeated DOYs should produce unique timestamps for SCOPE alignment."""
    ds = build_observation_dataset(
        doys=np.array([150, 150, 151]),
        year=2021,
        geojson_path=TEST_FIELD_GEOJSON,
    )
    times = ds.indexes["time"]
    assert times.is_unique
    assert str(times[0]) == "2021-05-30 10:30:00"
    assert str(times[1]) == "2021-05-30 10:35:00"


# ---------------------------------------------------------------------------
# bridge_arc_to_scope tests
# ---------------------------------------------------------------------------


def test_bridge_arc_to_scope_with_fixtures(sample_arc_outputs):
    """bridge_arc_to_scope should produce well-shaped DataArrays from an ArcResult."""
    out = sample_arc_outputs
    arc_result = ArcResult(
        scale_data=out["scale_data"],
        post_bio_tensor=out["post_bio_tensor"],
        post_bio_unc_tensor=out["post_bio_unc_tensor"],
        mask=out["mask"],
        doys=out["doys"],
        geotransform=out["geotransform"],
        crs=out["crs"],
    )
    bio_da, scale_da = bridge_arc_to_scope(arc_result, year=2021)
    assert isinstance(bio_da, xr.DataArray)
    assert isinstance(scale_da, xr.DataArray)
    assert bio_da.dims == ("y", "x", "band", "time")
    assert scale_da.dims == ("y", "x", "band")


def test_bridge_arc_to_scope_no_geotransform_raises(sample_arc_outputs):
    """bridge_arc_to_scope should raise when geotransform is None."""
    out = sample_arc_outputs
    arc_result = ArcResult(
        scale_data=out["scale_data"],
        post_bio_tensor=out["post_bio_tensor"],
        post_bio_unc_tensor=out["post_bio_unc_tensor"],
        mask=out["mask"],
        doys=out["doys"],
        geotransform=None,
        crs=out["crs"],
    )
    with pytest.raises(ValueError, match="geotransform"):
        bridge_arc_to_scope(arc_result, year=2021)


# ---------------------------------------------------------------------------
# fetch_weather tests
# ---------------------------------------------------------------------------


def test_fetch_weather_unknown_provider_raises():
    """An unknown weather_provider name should raise ValueError."""
    config = PipelineConfig(
        geojson_path=str(TEST_FIELD_GEOJSON),
        start_date="2021-05-15",
        end_date="2021-10-01",
        crop_type="wheat",
        start_of_season=170,
        year=2021,
        weather_provider="nonexistent_provider",
    )
    with pytest.raises(ValueError, match="Unknown weather provider"):
        fetch_weather(config)


def test_retrieve_arc_passes_data_source(tmp_path, monkeypatch):
    """retrieve_arc should forward config.data_source to ARC."""
    called: dict[str, object] = {}

    def fake_arc_field(**kwargs):
        called.update(kwargs)
        return (
            np.zeros((1, 15), dtype=np.float64),
            np.zeros((1, 7, 1), dtype=np.float64),
            np.zeros((1, 7, 1), dtype=np.float64),
            np.zeros((1, 1), dtype=bool),
            np.array([170], dtype=int),
        )

    monkeypatch.setitem(sys.modules, "arc", types.SimpleNamespace(arc_field=fake_arc_field))
    config = PipelineConfig(
        geojson_path=str(TEST_FIELD_GEOJSON),
        start_date="2021-05-15",
        end_date="2021-10-01",
        crop_type="wheat",
        start_of_season=170,
        year=2021,
        data_source="planetary",
        output_dir=tmp_path,
    )

    retrieve_arc(config)
    assert called["data_source"] == "planetary"
