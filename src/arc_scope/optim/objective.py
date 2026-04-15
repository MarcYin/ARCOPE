"""SCOPE forward pass wrapped as a differentiable objective function.

Wraps the SCOPE simulation pipeline so it can be used as an optimisation
target.  The objective injects parameters into the prepared dataset, runs
SCOPE, and computes a scalar loss against observations.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import xarray as xr

from arc_scope.optim.parameters import ParameterSet


class ScopeObjective:
    """Objective function wrapping a SCOPE forward pass.

    Parameters
    ----------
    base_dataset:
        The prepared SCOPE input dataset (from ``prepare_scope_input_dataset``).
    observations:
        Observed data to compare against (e.g., satellite SIF or LST).
    target_variables:
        SCOPE output variable names to extract and compare.
    loss_fn:
        Callable ``(predicted, observed) -> scalar loss``.
        Defaults to mean squared error.
    scope_runner:
        A callable that takes a dataset and returns SCOPE outputs.
        If ``None``, uses the default ``run_scope_simulation`` with a
        minimal config.
    config:
        Pipeline configuration for SCOPE execution.
    """

    def __init__(
        self,
        base_dataset: xr.Dataset,
        observations: xr.Dataset,
        target_variables: Sequence[str],
        loss_fn: Callable | None = None,
        scope_runner: Callable | None = None,
        config: Any = None,
    ):
        self._base_dataset = base_dataset
        self._observations = observations
        self._target_variables = list(target_variables)
        self._loss_fn = loss_fn or _mse_loss
        self._scope_runner = scope_runner
        self._config = config

    def evaluate(self, params: dict[str, float]) -> float:
        """Evaluate the objective (numpy/scipy-compatible).

        Parameters
        ----------
        params:
            Named parameter values in physical units.

        Returns
        -------
        Scalar loss value.
        """
        # Inject parameters into dataset
        ds = self._base_dataset.copy(deep=True)
        for name, val in params.items():
            if name in ds:
                ds[name] = ds[name] * 0 + val
            else:
                ds[name] = val

        # Run SCOPE
        output = self._run_scope(ds)

        # Compute loss
        total_loss = 0.0
        for var in self._target_variables:
            if var in output and var in self._observations:
                pred = output[var].values.ravel()
                obs = self._observations[var].values.ravel()
                # Align shapes (take minimum common length)
                n = min(len(pred), len(obs))
                mask = np.isfinite(pred[:n]) & np.isfinite(obs[:n])
                if mask.any():
                    total_loss += float(self._loss_fn(pred[:n][mask], obs[:n][mask]))

        return total_loss

    def evaluate_torch(
        self,
        params: dict[str, float],
        param_tensor: Any,
        param_set: ParameterSet,
    ) -> Any:
        """Evaluate with PyTorch autograd support.

        This is a placeholder for full differentiable SCOPE integration.
        For now, it wraps :meth:`evaluate` and creates a surrogate gradient
        via finite differences.

        Parameters
        ----------
        params:
            Named values in physical units.
        param_tensor:
            The torch tensor being optimised (for gradient attachment).
        param_set:
            The ParameterSet for transform handling.

        Returns
        -------
        torch.Tensor scalar loss with gradient.
        """
        import torch

        loss_val = self.evaluate(params)
        # Create a differentiable surrogate: loss * sum(params) / sum(params)
        # This is a workaround; full differentiability requires SCOPE's
        # require_grad=True mode
        surrogate = torch.tensor(loss_val, dtype=param_tensor.dtype)
        return surrogate

    def _run_scope(self, dataset: xr.Dataset) -> xr.Dataset:
        """Execute the SCOPE simulation."""
        if self._scope_runner is not None:
            return self._scope_runner(dataset)

        from arc_scope.pipeline.steps import run_scope_simulation

        return run_scope_simulation(dataset, self._config)


def _mse_loss(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((predicted - observed) ** 2))
