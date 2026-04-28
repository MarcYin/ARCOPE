"""Tests for the high-level pipeline runner."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import arc_scope.pipeline.runner as runner_module
import arc_scope.pipeline.steps as steps_module
from arc_scope.pipeline.config import PipelineConfig
from arc_scope.pipeline.optimization import OptimizationResult, optimization_enabled
from arc_scope.pipeline.runner import ArcScopePipeline
from arc_scope.pipeline.steps import ArcResult
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

    # Use integer index to avoid datetime encoding issues with scipy
    dummy_ds = xr.Dataset({
        "reflectance": (("time",), np.array([0.1, 0.2, 0.3])),
    }, coords={"time": np.arange(3)})

    output_path = pipeline._save_scope_output(dummy_ds)
    assert Path(output_path).exists()
    assert output_path.suffix == ".nc"

    # Verify roundtrip
    loaded = xr.open_dataset(output_path, engine="scipy")
    assert "reflectance" in loaded
    loaded.close()


def test_optimization_enabled_accepts_nested_runner_flag():
    """CPEO-style nested optim.enabled payloads should enable optimisation."""
    config = _make_config(optim_config={"optim": {"enabled": "true"}})

    assert optimization_enabled(config) is True


def test_run_executes_optimization_when_optim_config_enabled(monkeypatch, tmp_path):
    """optim_config.enabled should run a real optimisation before final output."""
    times = pd.date_range("2021-06-01", periods=3, freq="D")
    prepared_input = xr.Dataset(
        {"fqe": ("time", np.full(3, 0.01))},
        coords={"time": times},
    )
    observations = xr.Dataset(
        {"F740": ("time", np.full(3, 0.08))},
        coords={"time": times},
    )
    config = _make_config(
        scope_workflow="fluorescence",
        output_dir=tmp_path,
        save_scope_netcdf=False,
        optim_config={
            "enabled": True,
            "observations": observations,
            "target_variables": ["F740"],
            "parameters": [
                {
                    "name": "fqe",
                    "initial": 0.01,
                    "lower": 0.001,
                    "upper": 0.1,
                    "transform": "log",
                }
            ],
            "optimizer": {"type": "scipy", "use_autograd_jac": False},
            "max_iter": 80,
            "tol": 1e-10,
        },
    )
    arc_result = ArcResult(
        scale_data=np.zeros((1, 15)),
        post_bio_tensor=np.zeros((1, 7, 3)),
        post_bio_unc_tensor=np.zeros((1, 7, 3, 7)),
        mask=np.zeros((1, 1), dtype=bool),
        doys=np.array([152, 153, 154]),
    )

    monkeypatch.setattr(runner_module, "retrieve_arc", lambda cfg: arc_result)
    monkeypatch.setattr(
        runner_module,
        "bridge_arc_to_scope",
        lambda arc, year: (xr.DataArray([1.0]), xr.DataArray([1.0])),
    )
    monkeypatch.setattr(
        runner_module,
        "fetch_weather",
        lambda cfg: xr.Dataset({"Rin": ("time", np.full(3, 600.0))}, coords={"time": times}),
    )
    monkeypatch.setattr(
        runner_module,
        "build_observation_dataset",
        lambda doys, year, geojson_path: xr.Dataset(coords={"time": times}),
    )
    monkeypatch.setattr(
        runner_module,
        "prepare_scope_dataset",
        lambda post_bio, post_bio_scale, weather, observation, cfg: prepared_input.copy(deep=True),
    )
    def fake_run_scope_simulation(ds, cfg):
        return xr.Dataset({"F740": ds["fqe"] * 2.0})

    monkeypatch.setattr(runner_module, "run_scope_simulation", fake_run_scope_simulation)
    monkeypatch.setattr(steps_module, "run_scope_simulation", fake_run_scope_simulation)

    result = ArcScopePipeline(config).run()

    assert result.optimization_result is not None
    assert result.optimization_result.status == "optimized"
    assert result.optimization_result.optimized_loss < result.optimization_result.initial_loss
    assert result.optimization_result.parameters_optimized["fqe"] == pytest.approx(
        0.04,
        rel=1e-2,
    )
    np.testing.assert_allclose(result.scope_output_ds["F740"].values, 0.08, rtol=1e-2)
    assert result.scope_output_ds.attrs["arc_scope_optimization_status"] == "optimized"


def test_run_optimization_uses_pipeline_default_scope_runners(monkeypatch):
    """The high-level runner must not pass a lambda that disables built-in autograd."""
    captured = {}
    config = _make_config(scope_workflow="fluorescence", optimize=True)
    pipeline = ArcScopePipeline(config)

    def fake_run_pipeline_optimization(config, scope_input_ds, *, scope_runner=None):
        captured["scope_runner"] = scope_runner
        return (
            scope_input_ds,
            xr.Dataset({"F740": ("time", [0.08])}),
            OptimizationResult(
                status="optimized",
                target_variables=["F740"],
                initial_loss=1.0,
                optimized_loss=0.0,
                parameters_initial={"fqe": 0.01},
                parameters_optimized={"fqe": 0.04},
                optimizer="scipy:L-BFGS-B",
                converged=True,
            ),
        )

    monkeypatch.setattr(
        runner_module,
        "run_pipeline_optimization",
        fake_run_pipeline_optimization,
    )

    pipeline.run_optimization(xr.Dataset({"fqe": ("time", [0.01])}))

    assert captured["scope_runner"] is None


def test_run_optimization_requires_observations():
    """Optimisation must fail loudly instead of running an unoptimised simulation."""
    config = _make_config(scope_workflow="fluorescence", optimize=True)
    pipeline = ArcScopePipeline(config)

    with pytest.raises(ValueError, match="observed target data"):
        pipeline.run_optimization(xr.Dataset({"fqe": ("time", [0.01])}))
