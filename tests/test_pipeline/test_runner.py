"""Tests for the high-level pipeline runner."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arc_scope.pipeline.config import PipelineConfig
from arc_scope.pipeline.runner import ArcScopePipeline
from arc_scope.weather.base import REQUIRED_WEATHER_VARS


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> PipelineConfig:
    defaults = dict(
        geojson_path="/tmp/test.geojson",
        start_date="2021-05-15",
        end_date="2021-10-01",
        crop_type="wheat",
        start_of_season=170,
        year=2021,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_init():
    """ArcScopePipeline should accept a valid PipelineConfig without errors."""
    config = _make_config()
    pipeline = ArcScopePipeline(config)
    assert pipeline.config is config
    assert pipeline.config.year == 2021


def test_minimal_weather_shape():
    """_minimal_weather should return a dataset with the correct number of timesteps."""
    from arc_scope.pipeline.steps import ArcResult

    config = _make_config()
    pipeline = ArcScopePipeline(config)

    doys = np.array([150, 160, 170])
    arc_result = ArcResult(
        scale_data=np.zeros((1, 15)),
        post_bio_tensor=np.zeros((1, 7, 3)),
        post_bio_unc_tensor=np.zeros((1, 7, 3, 7)),
        mask=np.zeros((1, 1), dtype=bool),
        doys=doys,
    )

    ds = pipeline._minimal_weather(arc_result)
    assert ds.sizes["time"] == len(doys)


def test_minimal_weather_has_required_vars():
    """_minimal_weather should contain all REQUIRED_WEATHER_VARS."""
    from arc_scope.pipeline.steps import ArcResult

    config = _make_config()
    pipeline = ArcScopePipeline(config)

    doys = np.array([150, 160])
    arc_result = ArcResult(
        scale_data=np.zeros((1, 15)),
        post_bio_tensor=np.zeros((1, 7, 2)),
        post_bio_unc_tensor=np.zeros((1, 7, 2, 7)),
        mask=np.zeros((1, 1), dtype=bool),
        doys=doys,
    )

    ds = pipeline._minimal_weather(arc_result)
    for var in REQUIRED_WEATHER_VARS:
        assert var in ds, f"Minimal weather missing {var}"


def test_save_scope_output_creates_file(tmp_path):
    """_save_scope_output should write a NetCDF file to output_dir."""
    config = _make_config(output_dir=str(tmp_path / "results"))
    pipeline = ArcScopePipeline(config)

    dummy_ds = xr.Dataset({
        "reflectance": (("time",), np.array([0.1, 0.2, 0.3])),
    }, coords={"time": pd.date_range("2021-06-01", periods=3)})

    output_path = pipeline._save_scope_output(dummy_ds)
    assert Path(output_path).exists()
    assert output_path.suffix == ".nc"

    # Verify roundtrip
    loaded = xr.open_dataset(output_path)
    assert "reflectance" in loaded
    loaded.close()
