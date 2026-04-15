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
