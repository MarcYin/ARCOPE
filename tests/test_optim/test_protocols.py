"""Tests for optimisation protocol wrappers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from arc_scope.optim.objective import AutogradUnavailable
from arc_scope.optim.parameters import ParameterSet, ParameterSpec
from arc_scope.optim.protocols import ScipyOptimizer


def test_scipy_optimizer_passes_autograd_jac_to_minimize(monkeypatch):
    """ScipyOptimizer should provide an analytic jac callable when available."""
    calls: dict[str, object] = {}

    class Objective:
        def evaluate(self, params):
            return (params["x"] - 2.0) ** 2

        def evaluate_value_and_gradient(self, values, param_set):
            calls["gradient_x"] = np.array(values, copy=True)
            return 1.0, np.array([3.0])

    def fake_minimize(fun, x0, *, method, jac=None, options=None, tol=None):
        calls["method"] = method
        calls["x0"] = np.array(x0, copy=True)
        assert jac is not None
        assert fun(x0) == pytest.approx(1.0)
        np.testing.assert_allclose(jac(x0), [3.0])
        return SimpleNamespace(success=True, nit=1, x=np.array([2.0]))

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    params = ParameterSet([ParameterSpec("x", initial=1.0, lower=-10.0, upper=10.0)])
    optimizer = ScipyOptimizer(use_autograd_jac="auto")
    result = optimizer.step(Objective(), params)

    assert optimizer.converged()
    assert optimizer.used_autograd_jac is True
    assert result.specs[0].initial == pytest.approx(2.0)
    np.testing.assert_allclose(calls["x0"], [1.0])
    np.testing.assert_allclose(calls["gradient_x"], [1.0])


def test_scipy_optimizer_auto_falls_back_when_autograd_unavailable(monkeypatch):
    """Auto mode should preserve old scipy finite-difference behaviour."""
    jac_seen: list[bool] = []

    class Objective:
        def evaluate(self, params):
            return (params["x"] - 2.0) ** 2

        def evaluate_value_and_gradient(self, values, param_set):
            raise AutogradUnavailable("no differentiable forward path")

    def fake_minimize(fun, x0, *, method, jac=None, options=None, tol=None):
        jac_seen.append(jac is not None)
        if jac is not None:
            jac(x0)
        return SimpleNamespace(success=True, nit=1, x=np.array([2.0]))

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    params = ParameterSet([ParameterSpec("x", initial=1.0, lower=-10.0, upper=10.0)])
    optimizer = ScipyOptimizer(use_autograd_jac="auto")
    result = optimizer.step(Objective(), params)

    assert jac_seen == [True, False]
    assert optimizer.converged()
    assert optimizer.used_autograd_jac is False
    assert "no differentiable forward path" in str(optimizer.autograd_jac_error)
    assert result.specs[0].initial == pytest.approx(2.0)


def test_scipy_optimizer_required_autograd_raises(monkeypatch):
    """Required mode should fail instead of silently finite-differencing."""

    class Objective:
        def evaluate(self, params):
            return (params["x"] - 2.0) ** 2

        def evaluate_value_and_gradient(self, values, param_set):
            raise AutogradUnavailable("no differentiable forward path")

    def fake_minimize(fun, x0, *, method, jac=None, options=None, tol=None):
        assert jac is not None
        jac(x0)

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    params = ParameterSet([ParameterSpec("x", initial=1.0, lower=-10.0, upper=10.0)])
    optimizer = ScipyOptimizer(use_autograd_jac="required")

    with pytest.raises(AutogradUnavailable, match="no differentiable"):
        optimizer.step(Objective(), params)


def test_scipy_optimizer_can_disable_autograd_jac(monkeypatch):
    """Explicit False should keep scipy's finite-difference path."""
    jac_values: list[object] = []

    class Objective:
        def evaluate(self, params):
            return (params["x"] - 2.0) ** 2

    def fake_minimize(fun, x0, *, method, jac=None, options=None, tol=None):
        jac_values.append(jac)
        return SimpleNamespace(success=True, nit=1, x=np.array([2.0]))

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    params = ParameterSet([ParameterSpec("x", initial=1.0, lower=-10.0, upper=10.0)])
    ScipyOptimizer(use_autograd_jac=False).step(Objective(), params)

    assert jac_values == [None]


def test_scipy_optimizer_string_false_disables_autograd_jac(monkeypatch):
    """String config values from runner payloads should disable the jac hook."""
    jac_values: list[object] = []

    class Objective:
        def evaluate(self, params):
            return (params["x"] - 2.0) ** 2

    def fake_minimize(fun, x0, *, method, jac=None, options=None, tol=None):
        jac_values.append(jac)
        return SimpleNamespace(success=True, nit=1, x=np.array([2.0]))

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    params = ParameterSet([ParameterSpec("x", initial=1.0, lower=-10.0, upper=10.0)])
    ScipyOptimizer(use_autograd_jac="false").step(Objective(), params)

    assert jac_values == [None]
