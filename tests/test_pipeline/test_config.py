"""Tests for pipeline configuration."""
from __future__ import annotations

from pathlib import Path

import pytest

from arc_scope.pipeline.config import PipelineConfig, WORKFLOW_OPTIONS


def test_valid_config():
    config = PipelineConfig(
        geojson_path="/tmp/test.geojson",
        start_date="2021-05-15",
        end_date="2021-10-01",
        crop_type="wheat",
        start_of_season=170,
        year=2021,
    )
    assert config.scope_workflow == "reflectance"
    assert isinstance(config.geojson_path, Path)


def test_invalid_workflow():
    with pytest.raises(ValueError, match="Unknown workflow"):
        PipelineConfig(
            geojson_path="/tmp/test.geojson",
            start_date="2021-05-15",
            end_date="2021-10-01",
            crop_type="wheat",
            start_of_season=170,
            year=2021,
            scope_workflow="invalid",
        )


def test_resolved_scope_options():
    config = PipelineConfig(
        geojson_path="/tmp/test.geojson",
        start_date="2021-05-15",
        end_date="2021-10-01",
        crop_type="wheat",
        start_of_season=170,
        year=2021,
        scope_workflow="fluorescence",
        scope_options={"calc_directional": 1},
    )
    opts = config.resolved_scope_options
    assert opts["calc_fluor"] == 1
    assert opts["calc_planck"] == 0
    assert opts["calc_directional"] == 1  # User override


# ---------------------------------------------------------------------------
# Additional tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workflow", list(WORKFLOW_OPTIONS.keys()))
def test_all_workflow_types(workflow):
    """Every workflow listed in WORKFLOW_OPTIONS should be accepted by PipelineConfig."""
    config = PipelineConfig(
        geojson_path="/tmp/test.geojson",
        start_date="2021-05-15",
        end_date="2021-10-01",
        crop_type="wheat",
        start_of_season=170,
        year=2021,
        scope_workflow=workflow,
    )
    assert config.scope_workflow == workflow
    opts = config.resolved_scope_options
    assert "calc_fluor" in opts
    assert "calc_planck" in opts


def test_output_dir_becomes_path():
    """output_dir should be converted to a Path regardless of input type."""
    config = PipelineConfig(
        geojson_path="/tmp/test.geojson",
        start_date="2021-05-15",
        end_date="2021-10-01",
        crop_type="wheat",
        start_of_season=170,
        year=2021,
        output_dir="/tmp/my_output",
    )
    assert isinstance(config.output_dir, Path)
    assert str(config.output_dir) == "/tmp/my_output"


def test_default_values():
    """Default field values should match expected baseline settings."""
    config = PipelineConfig(
        geojson_path="/tmp/test.geojson",
        start_date="2021-05-15",
        end_date="2021-10-01",
        crop_type="wheat",
        start_of_season=170,
        year=2021,
    )
    assert config.num_samples == 100000
    assert config.growth_season_length == 45
    assert config.data_source == "aws"
    assert config.weather_provider == "era5"
    assert config.scope_workflow == "reflectance"
    assert config.device == "cpu"
    assert config.dtype == "float64"
    assert config.save_arc_npz is True
    assert config.save_scope_netcdf is True
    assert config.optimize is False
