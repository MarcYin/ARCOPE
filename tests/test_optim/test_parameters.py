"""Tests for optimisation parameter containers."""

import numpy as np
import pytest

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
