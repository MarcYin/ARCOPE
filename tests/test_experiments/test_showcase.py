"""Tests for the core-only showcase experiment."""

from __future__ import annotations

import json

import pytest

from arc_scope.experiments import run_showcase_experiment, write_showcase_artifacts
from arc_scope.experiments.showcase import main


def test_run_showcase_experiment_shapes_and_metrics():
    """The showcase experiment should return aligned datasets and improve fit."""
    result = run_showcase_experiment(seed=7)

    assert result.post_bio_da.sizes["time"] == result.summary.n_time_steps
    assert result.weather_ds.sizes["time"] == result.summary.n_time_steps
    assert result.observation_ds.sizes["time"] == result.summary.n_time_steps
    assert result.experiment_ds.sizes["time"] == result.summary.n_time_steps
    assert list(result.timeseries.columns[:4]) == ["date", "lai", "cab", "cw"]

    assert result.summary.peak_lai > 4.0
    assert 0.0 < result.summary.mean_direct_fraction <= 1.0
    assert 0.0 < result.summary.mean_diffuse_fraction <= 1.0
    assert result.summary.mean_direct_fraction + result.summary.mean_diffuse_fraction == pytest.approx(1.0)
    assert result.summary.rmse_optimized < result.summary.rmse_initial
    assert result.summary.relative_fqe_error_pct < 15.0


def test_write_showcase_artifacts(tmp_path):
    """Docs-ready CSV, JSON, and SVG artifacts should be written to disk."""
    result = run_showcase_experiment(seed=7)
    files = write_showcase_artifacts(result, tmp_path)

    for path in files.values():
        assert path.exists()
        assert path.stat().st_size > 0

    summary = json.loads(files["summary"].read_text(encoding="utf-8"))
    assert "optimized_fqe" in summary
    assert "rmse_optimized" in summary


def test_showcase_module_cli(tmp_path):
    """The packaged module entry point should work without the repo example file."""
    main(["--output-dir", str(tmp_path)])
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "timeseries.csv").exists()
