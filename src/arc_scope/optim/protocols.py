"""Pluggable optimiser protocol and built-in wrappers.

The ``Optimizer`` protocol defines the interface that any optimiser must
satisfy.  This allows gradient-based (PyTorch), gradient-free (scipy),
and custom Bayesian optimisers to be used interchangeably.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

import numpy as np

from arc_scope.optim.parameters import ParameterSet

if TYPE_CHECKING:
    from arc_scope.optim.objective import ScopeObjective


@runtime_checkable
class Optimizer(Protocol):
    """Protocol for SCOPE parameter optimisers."""

    def step(
        self,
        objective: ScopeObjective,
        params: ParameterSet,
    ) -> ParameterSet:
        """Perform one optimisation step.

        Parameters
        ----------
        objective:
            The SCOPE forward-pass objective function.
        params:
            Current parameter values.

        Returns
        -------
        Updated ParameterSet with improved values.
        """
        ...

    def converged(self) -> bool:
        """Whether the optimiser has reached convergence."""
        ...


class ScipyOptimizer:
    """Wrapper around ``scipy.optimize.minimize`` for gradient-free optimisation.

    Parameters
    ----------
    method:
        Scipy minimisation method (default ``"L-BFGS-B"``).
    max_iter:
        Maximum number of iterations.
    tol:
        Convergence tolerance.
    """

    def __init__(
        self,
        method: str = "L-BFGS-B",
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        self._method = method
        self._max_iter = max_iter
        self._tol = tol
        self._converged = False
        self._n_iter = 0

    def step(self, objective: ScopeObjective, params: ParameterSet) -> ParameterSet:
        """Run scipy minimisation from the current parameter values."""
        from scipy.optimize import minimize

        x0 = params.to_array()

        def _obj(x: np.ndarray) -> float:
            named = params.from_array(x)
            return float(objective.evaluate(named))

        result = minimize(
            _obj,
            x0,
            method=self._method,
            options={"maxiter": self._max_iter},
            tol=self._tol,
        )

        self._converged = result.success
        self._n_iter += result.nit

        # Update parameter specs with optimised values
        optimised = params.from_array(result.x)
        for spec in params.specs:
            if spec.optimize and spec.name in optimised:
                spec.initial = optimised[spec.name]

        return params

    def converged(self) -> bool:
        return self._converged


class TorchOptimizer:
    """Wrapper around ``torch.optim`` optimisers for gradient-based optimisation.

    Parameters
    ----------
    optimizer_cls:
        A ``torch.optim.Optimizer`` class (e.g., ``torch.optim.Adam``).
    lr:
        Learning rate.
    max_steps:
        Maximum gradient steps per call to :meth:`step`.
    tol:
        Convergence tolerance on loss change.
    optimizer_kwargs:
        Extra keyword arguments for the PyTorch optimiser.
    """

    def __init__(
        self,
        optimizer_cls: type | None = None,
        lr: float = 0.01,
        max_steps: int = 100,
        tol: float = 1e-6,
        **optimizer_kwargs,
    ):
        self._optimizer_cls = optimizer_cls
        self._lr = lr
        self._max_steps = max_steps
        self._tol = tol
        self._optimizer_kwargs = optimizer_kwargs
        self._converged = False

    def step(self, objective: ScopeObjective, params: ParameterSet) -> ParameterSet:
        """Run gradient descent from the current parameter values."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required. Install with: pip install arc-scope[optim]")

        optimizer_cls = self._optimizer_cls or torch.optim.Adam
        param_tensor = params.to_torch(device="cpu")
        optimizer = optimizer_cls([param_tensor], lr=self._lr, **self._optimizer_kwargs)

        prev_loss = float("inf")
        for i in range(self._max_steps):
            optimizer.zero_grad()
            named = params.from_array(param_tensor.detach().cpu().numpy())
            loss = objective.evaluate_torch(named, param_tensor, params)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if abs(prev_loss - current_loss) < self._tol:
                self._converged = True
                break
            prev_loss = current_loss

        # Update specs with optimised values
        optimised = params.from_array(param_tensor.detach().cpu().numpy())
        for spec in params.specs:
            if spec.optimize and spec.name in optimised:
                spec.initial = optimised[spec.name]

        return params

    def converged(self) -> bool:
        return self._converged
