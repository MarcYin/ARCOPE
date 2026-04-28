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


class AutogradUnavailable(RuntimeError):
    """Raised when an objective cannot provide an autograd gradient."""


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

        missing_from_output = [var for var in self._target_variables if var not in output]
        if missing_from_output:
            raise ValueError(
                "SCOPE output is missing target variables: "
                + ", ".join(missing_from_output)
            )
        missing_from_observations = [
            var for var in self._target_variables if var not in self._observations
        ]
        if missing_from_observations:
            raise ValueError(
                "Observations are missing target variables: "
                + ", ".join(missing_from_observations)
            )

        # Compute loss
        total_loss = 0.0
        compared = False
        for var in self._target_variables:
            pred = output[var].values.ravel()
            obs = self._observations[var].values.ravel()
            # Align shapes (take minimum common length)
            n = min(len(pred), len(obs))
            mask = np.isfinite(pred[:n]) & np.isfinite(obs[:n])
            if mask.any():
                compared = True
                total_loss += float(self._loss_fn(pred[:n][mask], obs[:n][mask]))

        if not compared:
            raise ValueError("Objective has no finite prediction/observation pairs.")

        return total_loss

    def evaluate_torch(
        self,
        params: dict[str, float],
        param_tensor: Any,
        param_set: ParameterSet,
    ) -> Any:
        """Evaluate with PyTorch autograd support.

        This path keeps the parameter transform and objective framing in
        ARC-SCOPE while allowing a PyTorch-backed SCOPE forward pass to
        contribute gradients to scipy optimisers.

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

        torch_params = param_set.from_torch(param_tensor)
        loss = self._evaluate_torch_params(
            torch_params,
            torch=torch,
            dtype=param_tensor.dtype,
            device=param_tensor.device,
        )
        if not bool(getattr(loss, "requires_grad", False)):
            raise AutogradUnavailable(
                "Objective loss is not connected to the optimisation tensor. "
                "Use a PyTorch-backed scope_runner and differentiable loss."
            )
        return loss

    def evaluate_value_and_gradient(
        self,
        values: np.ndarray,
        param_set: ParameterSet,
    ) -> tuple[float, np.ndarray]:
        """Evaluate loss and autograd gradient in unconstrained parameter space."""
        try:
            import torch
        except ImportError as exc:
            raise AutogradUnavailable(
                "PyTorch is required for scipy autograd gradients."
            ) from exc

        tensor = torch.tensor(
            np.asarray(values, dtype=np.float64),
            dtype=torch.float64,
            requires_grad=True,
        )
        try:
            loss = self.evaluate_torch({}, tensor, param_set)
        except AutogradUnavailable:
            raise
        except (AttributeError, RuntimeError, TypeError) as exc:
            raise AutogradUnavailable(
                "Autograd objective evaluation failed. Falling back to scipy "
                "finite differences is allowed when ScipyOptimizer uses "
                "use_autograd_jac='auto'."
            ) from exc

        loss.backward()
        if tensor.grad is None:
            raise AutogradUnavailable("Autograd did not produce a parameter gradient.")
        gradient = tensor.grad.detach().cpu().numpy().astype(np.float64, copy=False)
        return float(loss.detach().cpu().item()), gradient

    def _run_scope(self, dataset: xr.Dataset) -> xr.Dataset:
        """Execute the SCOPE simulation."""
        if self._scope_runner is not None:
            return self._scope_runner(dataset)

        from arc_scope.pipeline.steps import run_scope_simulation

        return run_scope_simulation(dataset, self._config)

    def _evaluate_torch_params(
        self,
        params: dict[str, Any],
        *,
        torch: Any,
        dtype: Any,
        device: Any,
    ) -> Any:
        ds = self._inject_params(params)
        output = self._run_scope(ds)
        return self._torch_loss(
            output,
            torch=torch,
            dtype=dtype,
            device=device,
        )

    def _inject_params(self, params: dict[str, Any]) -> xr.Dataset:
        ds = self._base_dataset.copy(deep=True)
        for name, val in params.items():
            if _is_torch_tensor(val):
                ds[name] = _torch_dataarray_like(ds.get(name), val)
            elif name in ds:
                ds[name] = ds[name] * 0 + val
            else:
                ds[name] = val
        return ds

    def _torch_loss(
        self,
        output: xr.Dataset,
        *,
        torch: Any,
        dtype: Any,
        device: Any,
    ) -> Any:
        missing_from_output = [var for var in self._target_variables if var not in output]
        if missing_from_output:
            raise ValueError(
                "SCOPE output is missing target variables: "
                + ", ".join(missing_from_output)
            )
        missing_from_observations = [
            var for var in self._target_variables if var not in self._observations
        ]
        if missing_from_observations:
            raise ValueError(
                "Observations are missing target variables: "
                + ", ".join(missing_from_observations)
            )

        total_loss = None
        compared = False
        for var in self._target_variables:
            pred = _as_torch_tensor(output[var], torch=torch, dtype=dtype, device=device)
            obs = _as_torch_tensor(
                self._observations[var],
                torch=torch,
                dtype=dtype,
                device=device,
            )
            pred = pred.reshape(-1)
            obs = obs.reshape(-1)
            n = min(int(pred.numel()), int(obs.numel()))
            mask = torch.isfinite(pred[:n]) & torch.isfinite(obs[:n])
            if bool(mask.any().detach().cpu().item()):
                compared = True
                pred_valid = pred[:n][mask]
                obs_valid = obs[:n][mask]
                loss = self._torch_target_loss(pred_valid, obs_valid, torch=torch)
                total_loss = loss if total_loss is None else total_loss + loss

        if not compared or total_loss is None:
            raise ValueError("Objective has no finite prediction/observation pairs.")
        return total_loss

    def _torch_target_loss(self, predicted: Any, observed: Any, *, torch: Any) -> Any:
        if self._loss_fn is _mse_loss:
            return torch.mean((predicted - observed) ** 2)
        loss = self._loss_fn(predicted, observed)
        if _is_torch_tensor(loss):
            return loss
        return torch.as_tensor(loss, dtype=predicted.dtype, device=predicted.device)


def _mse_loss(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((predicted - observed) ** 2))


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and value.__class__.__name__ == "Tensor"


def _torch_dataarray_like(template: xr.DataArray | None, value: Any) -> xr.DataArray:
    if template is None:
        return xr.DataArray(value.reshape(()))
    shape = tuple(template.shape)
    if shape:
        data = value * value.new_ones(shape)
    else:
        data = value.reshape(())
    return xr.DataArray(
        data,
        dims=template.dims,
        coords=template.coords,
        attrs=template.attrs,
        name=template.name,
    )


def _as_torch_tensor(value: Any, *, torch: Any, dtype: Any, device: Any) -> Any:
    if isinstance(value, xr.DataArray):
        value = value.data
    if _is_torch_tensor(value):
        return value.to(dtype=dtype, device=device)
    return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)
