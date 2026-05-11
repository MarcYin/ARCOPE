"""Tests for optimisation parameter containers."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arc_scope.optim.parameters import ParameterSet, ParameterSpec


def test_parameter_spec_identity_transform():
    spec = ParameterSpec("x", initial=5.0, lower=0.0, upper=10.0, transform="identity")
    assert spec.to_unconstrained(5.0) == 5.0
    assert spec.to_physical(5.0) == 5.0


def test_parameter_spec_log_transform():
    spec = ParameterSpec("x", initial=1.0, lower=0.1, upper=100.0, transform="log")
    unc = spec.to_unconstrained(1.0)
    assert unc == pytest.approx(0.0, abs=1e-10)
    assert spec.to_physical(unc) == pytest.approx(1.0, abs=1e-6)


def test_parameter_spec_logit_transform():
    spec = ParameterSpec("x", initial=0.5, lower=0.0, upper=1.0, transform="logit")
    unc = spec.to_unconstrained(0.5)
    assert unc == pytest.approx(0.0, abs=1e-6)
    recovered = spec.to_physical(unc)
    assert recovered == pytest.approx(0.5, abs=1e-6)


def test_parameter_set_to_array():
    params = ParameterSet([
        ParameterSpec("a", initial=1.0, lower=0.0, upper=10.0),
        ParameterSpec("b", initial=2.0, lower=0.0, upper=10.0, optimize=False),
        ParameterSpec("c", initial=3.0, lower=0.0, upper=10.0),
    ])
    arr = params.to_array()
    assert len(arr) == 2  # Only optimizable params
    assert arr[0] == pytest.approx(1.0)
    assert arr[1] == pytest.approx(3.0)


def test_parameter_set_from_array():
    params = ParameterSet([
        ParameterSpec("a", initial=1.0, lower=0.0, upper=10.0),
        ParameterSpec("b", initial=2.0, lower=0.0, upper=10.0, optimize=False),
    ])
    result = params.from_array(np.array([5.0]))
    assert result["a"] == pytest.approx(5.0)
    assert result["b"] == pytest.approx(2.0)  # Fixed


def test_parameter_set_roundtrip():
    params = ParameterSet([
        ParameterSpec("fqe", initial=0.01, lower=0.001, upper=0.1, transform="log"),
    ])
    arr = params.to_array()
    result = params.from_array(arr)
    assert result["fqe"] == pytest.approx(0.01, rel=1e-5)


def test_parameter_set_injects_missing_parameter_on_scope_grid():
    ds = xr.Dataset(
        {
            "lai": (
                ("y", "x", "time"),
                np.ones((2, 3, 4), dtype=float),
            ),
        },
        coords={
            "y": [10.0, 20.0],
            "x": [100.0, 200.0, 300.0],
            "time": pd.date_range("2021-06-01", periods=4),
        },
    )
    params = ParameterSet([
        ParameterSpec("rss", initial=500.0, lower=10.0, upper=5000.0),
    ])

    injected = params.inject_into_dataset(ds)

    assert injected["rss"].dims == ("y", "x", "time")
    assert injected["rss"].shape == (2, 3, 4)
    np.testing.assert_allclose(injected["rss"].values, 500.0)
    np.testing.assert_array_equal(injected["rss"].coords["y"], ds.coords["y"])
    np.testing.assert_array_equal(injected["rss"].coords["x"], ds.coords["x"])
    np.testing.assert_array_equal(injected["rss"].coords["time"], ds.coords["time"])
