"""Tests for generated optimisation example artifacts."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from arc_scope.experiments import generate_optimization_examples


def test_generate_optimization_examples_writes_fit_artifacts(tmp_path):
    """Docs optimisation examples should emit concrete tables and SVG plots."""
    pytest.importorskip("matplotlib")

    files = generate_optimization_examples(tmp_path)

    for path in files.values():
        assert path.exists()
        assert path.stat().st_size > 0

    summary = json.loads(files["summary"].read_text(encoding="utf-8"))
    assert set(summary) == {"sif", "thermal", "energy_balance"}

    for case in summary.values():
        assert case["converged"] is True
        assert case["optimized_loss"] < case["initial_loss"]
        assert case["parameters_optimized"] != case["parameters_initial"]

    timeseries = pd.read_csv(files["timeseries"])
    assert {"scenario", "time", "target", "observed", "initial", "optimized"} <= set(
        timeseries.columns
    )
    assert set(timeseries["scenario"]) == {"sif", "thermal", "energy_balance"}

    parameters = pd.read_csv(files["parameters"])
    assert {"scenario", "parameter", "initial", "optimized", "true"} <= set(
        parameters.columns
    )
