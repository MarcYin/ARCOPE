"""Tests for the SCOPE objective function and loss utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arc_scope.optim.objective import ScopeObjective, _mse_loss
from arc_scope.optim.parameters import ParameterSet, ParameterSpec


# ---------------------------------------------------------------------------
# _mse_loss tests
# ---------------------------------------------------------------------------


def test_mse_loss_zero_identical():
    """MSE of two identical arrays should be exactly 0."""
    a = np.array([1.0, 2.0, 3.0])
    assert _mse_loss(a, a) == pytest.approx(0.0, abs=1e-15)


def test_mse_loss_nonzero():
    """MSE of known differing arrays should match the hand-calculated value."""
    predicted = np.array([1.0, 2.0, 3.0])
    observed = np.array([1.0, 3.0, 5.0])
    # differences: 0, 1, 2 -> squares: 0, 1, 4 -> mean = 5/3
    expected = 5.0 / 3.0
    assert _mse_loss(predicted, observed) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# ScopeObjective tests
# ---------------------------------------------------------------------------


def test_scope_objective_init():
    """ScopeObjective should initialise without errors given minimal inputs."""
    base_ds = xr.Dataset({"lai": ("time", [3.0, 4.0])})
    obs_ds = xr.Dataset({"sif": ("time", [0.5, 0.6])})

    obj = ScopeObjective(
        base_dataset=base_ds,
        observations=obs_ds,
        target_variables=["sif"],
    )
    assert obj._target_variables == ["sif"]
    # The default loss function should be _mse_loss
    assert obj._loss_fn is _mse_loss


def test_scope_objective_raises_for_missing_output_target():
    """Missing target variables should not silently produce a zero loss."""
    base_ds = xr.Dataset({"fqe": ("time", [0.01, 0.02])})
    obs_ds = xr.Dataset({"sif": ("time", [0.5, 0.6])})
    obj = ScopeObjective(
        base_dataset=base_ds,
        observations=obs_ds,
        target_variables=["sif"],
        scope_runner=lambda ds: xr.Dataset({"other": ("time", [1.0, 2.0])}),
    )

    with pytest.raises(ValueError, match="missing target variables"):
        obj.evaluate({"fqe": 0.01})


def test_scope_objective_requires_selector_for_spatial_prediction_point_observation():
    """Gridded predictions must not be truncated against point observations."""
    times = pd.date_range("2021-06-01", periods=4, freq="D")
    pred = xr.Dataset(
        {
            "target": (
                ("y", "x", "time"),
                np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4),
            )
        },
        coords={"y": [10, 11], "x": [20, 21, 22], "time": times},
    )
    obs = xr.Dataset(
        {"target": ("time", np.full(4, 999.0))},
        coords={"time": times},
    )
    obj = ScopeObjective(
        base_dataset=pred,
        observations=obs,
        target_variables=["target"],
        scope_runner=lambda ds: pred,
    )

    with pytest.raises(ValueError, match="extra dims"):
        obj.evaluate({})


def test_scope_objective_pixel_selector_aligns_spatial_prediction_by_time():
    """A point observation should compare with the selected pixel time series."""
    times = pd.date_range("2021-06-01", periods=4, freq="D")
    values = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    pred = xr.Dataset(
        {"target": (("y", "x", "time"), values)},
        coords={"y": [10, 11], "x": [20, 21, 22], "time": times},
    )
    observed = values[1, 2, :] + np.array([1.0, -1.0, 2.0, -2.0])
    obs = xr.Dataset(
        {"target": ("time", observed)},
        coords={"time": times},
    )
    obj = ScopeObjective(
        base_dataset=pred,
        observations=obs,
        target_variables=["target"],
        scope_runner=lambda ds: pred,
        pixel_selector={"y": 11, "x": 22},
    )

    expected = np.mean((values[1, 2, :] - observed) ** 2)
    assert obj.evaluate({}) == pytest.approx(expected)


def test_scope_objective_raises_for_non_overlapping_time_coordinates():
    """Matching shapes are not enough when coordinate labels do not overlap."""
    pred = xr.Dataset(
        {"target": ("time", [1.0, 2.0, 3.0])},
        coords={"time": pd.date_range("2021-06-01", periods=3, freq="D")},
    )
    obs = xr.Dataset(
        {"target": ("time", [1.0, 2.0, 3.0])},
        coords={"time": pd.date_range("2022-06-01", periods=3, freq="D")},
    )
    obj = ScopeObjective(
        base_dataset=pred,
        observations=obs,
        target_variables=["target"],
        scope_runner=lambda ds: pred,
    )

    with pytest.raises(ValueError, match="overlapping coordinates"):
        obj.evaluate({})


# ---------------------------------------------------------------------------
# ParameterSet.inject_into_dataset test
# ---------------------------------------------------------------------------


def test_parameter_set_inject_into_dataset():
    """inject_into_dataset should set or overwrite variables in the dataset."""
    ds = xr.Dataset({
        "fqe": ("time", [0.01, 0.01, 0.01]),
        "lai": ("time", [3.0, 4.0, 5.0]),
    }, coords={"time": pd.date_range("2021-06-01", periods=3)})

    params = ParameterSet([
        ParameterSpec("fqe", initial=0.05, lower=0.001, upper=0.1),
        ParameterSpec("rss", initial=500.0, lower=10.0, upper=5000.0),
    ])

    modified = params.inject_into_dataset(ds)
    # fqe should be overwritten with 0.05 (broadcast to shape)
    np.testing.assert_allclose(modified["fqe"].values, 0.05)
    # rss was not in the original dataset, so it should be added as a scalar
    assert "rss" in modified
    assert float(modified["rss"]) == pytest.approx(500.0)
    # lai should remain unchanged
    np.testing.assert_array_equal(modified["lai"].values, [3.0, 4.0, 5.0])
