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
    torch_scope_runner:
        Optional callable used only by autograd evaluations. It receives the
        base dataset and the named PyTorch parameter tensors, and should return
        SCOPE outputs as torch tensors or torch-backed xarray variables without
        converting through NumPy.
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
        torch_scope_runner: Callable | None = None,
        config: Any = None,
    ):
        self._base_dataset = base_dataset
        self._observations = observations
        self._target_variables = list(target_variables)
        self._loss_fn = loss_fn or _mse_loss
        self._scope_runner = scope_runner
        self._torch_scope_runner = torch_scope_runner
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
        if self._can_stream_torch_loss():
            try:
                tensor.grad = None
                value = self._backward_torch_streaming(tensor, param_set, torch=torch)
            except AutogradUnavailable:
                raise
            except (AttributeError, RuntimeError, TypeError) as exc:
                raise AutogradUnavailable(
                    "Streaming autograd objective evaluation failed. Falling back "
                    "to scipy finite differences is allowed when ScipyOptimizer "
                    "uses use_autograd_jac='auto'."
                ) from exc
            if tensor.grad is None:
                raise AutogradUnavailable(
                    "Autograd did not produce a parameter gradient."
                )
            gradient = tensor.grad.detach().cpu().numpy().astype(np.float64, copy=False)
            return value, gradient

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

    def backward_torch_streaming(
        self,
        params: dict[str, float],
        param_tensor: Any,
        param_set: ParameterSet,
    ) -> float | None:
        """Backpropagate a streamed chunk loss when the runner supports it.

        Returns the scalar loss value when streaming was used, otherwise
        ``None`` so callers can keep the regular full-batch loss path.
        """
        if not self._can_stream_torch_loss():
            return None

        import torch

        _ = params
        return self._backward_torch_streaming(param_tensor, param_set, torch=torch)

    def _run_scope(self, dataset: xr.Dataset) -> xr.Dataset:
        """Execute the SCOPE simulation."""
        if self._scope_runner is not None:
            return self._scope_runner(dataset)

        from arc_scope.pipeline.steps import run_scope_simulation

        return run_scope_simulation(dataset, self._config)

    def _run_scope_torch(self, dataset: xr.Dataset, params: dict[str, Any]) -> Any:
        """Execute the differentiable SCOPE simulation path."""
        if self._torch_scope_runner is not None:
            return self._torch_scope_runner(dataset, params)
        return self._run_scope(self._inject_params(params))

    def _evaluate_torch_params(
        self,
        params: dict[str, Any],
        *,
        torch: Any,
        dtype: Any,
        device: Any,
    ) -> Any:
        output = self._run_scope_torch(self._base_dataset, params)
        return self._torch_loss(
            output,
            torch=torch,
            dtype=dtype,
            device=device,
        )

    def _can_stream_torch_loss(self) -> bool:
        return (
            self._loss_fn is _mse_loss
            and self._torch_scope_runner is not None
            and callable(getattr(self._torch_scope_runner, "iter_chunks", None))
        )

    def _backward_torch_streaming(
        self,
        param_tensor: Any,
        param_set: ParameterSet,
        *,
        torch: Any,
    ) -> float:
        torch_params = param_set.from_torch(param_tensor)
        dtype = param_tensor.dtype
        device = param_tensor.device
        observation_tensors = {
            var: _as_torch_tensor(
                self._observations[var],
                torch=torch,
                dtype=dtype,
                device=device,
            ).reshape(-1)
            for var in self._target_variables
        }

        with torch.no_grad():
            counts, total_sse = self._streaming_mse_totals(
                torch_params,
                observation_tensors,
                torch=torch,
                dtype=dtype,
                device=device,
            )
        if not counts:
            raise ValueError("Objective has no finite prediction/observation pairs.")

        total_loss_value = sum(total_sse[var] / counts[var] for var in counts)
        self._backward_streaming_mse(
            torch_params,
            observation_tensors,
            counts,
            torch=torch,
            dtype=dtype,
            device=device,
        )
        return float(total_loss_value)

    def _streaming_mse_totals(
        self,
        params: dict[str, Any],
        observation_tensors: dict[str, Any],
        *,
        torch: Any,
        dtype: Any,
        device: Any,
    ) -> tuple[dict[str, int], dict[str, float]]:
        counts = {var: 0 for var in self._target_variables}
        total_sse = {var: 0.0 for var in self._target_variables}
        offsets = {var: 0 for var in self._target_variables}
        for output in self._iter_scope_torch_chunks(self._base_dataset, params):
            self._validate_torch_output(output)
            for var in self._target_variables:
                pred, obs, offsets[var] = self._aligned_streaming_tensors(
                    output[var],
                    observation_tensors[var],
                    offsets[var],
                    torch=torch,
                    dtype=dtype,
                    device=device,
                )
                mask = torch.isfinite(pred) & torch.isfinite(obs)
                if bool(mask.any().detach().cpu().item()):
                    diff = pred[mask] - obs[mask]
                    counts[var] += int(mask.sum().detach().cpu().item())
                    total_sse[var] += float(torch.sum(diff * diff).detach().cpu().item())

        compared_counts = {
            var: count for var, count in counts.items() if count > 0
        }
        compared_sse = {
            var: total_sse[var] for var in compared_counts
        }
        return compared_counts, compared_sse

    def _backward_streaming_mse(
        self,
        params: dict[str, Any],
        observation_tensors: dict[str, Any],
        counts: dict[str, int],
        *,
        torch: Any,
        dtype: Any,
        device: Any,
    ) -> None:
        offsets = {var: 0 for var in self._target_variables}
        for output in self._iter_scope_torch_chunks(self._base_dataset, params):
            self._validate_torch_output(output)
            chunk_loss = None
            for var in self._target_variables:
                pred, obs, offsets[var] = self._aligned_streaming_tensors(
                    output[var],
                    observation_tensors[var],
                    offsets[var],
                    torch=torch,
                    dtype=dtype,
                    device=device,
                )
                mask = torch.isfinite(pred) & torch.isfinite(obs)
                if bool(mask.any().detach().cpu().item()) and counts.get(var, 0) > 0:
                    diff = pred[mask] - obs[mask]
                    loss = torch.sum(diff * diff) / counts[var]
                    chunk_loss = loss if chunk_loss is None else chunk_loss + loss
            if chunk_loss is not None:
                # The transformed physical parameters are shared across chunk
                # forwards. Retaining that tiny upstream graph avoids requiring
                # one full graph over all SCOPE chunks.
                chunk_loss.backward(retain_graph=True)
            del output

    def _iter_scope_torch_chunks(self, dataset: xr.Dataset, params: dict[str, Any]):
        iter_chunks = getattr(self._torch_scope_runner, "iter_chunks", None)
        if iter_chunks is None:
            raise AutogradUnavailable(
                "The configured SCOPE runner does not expose chunk streaming."
            )
        return iter_chunks(dataset, params)

    def _validate_torch_output(self, output: Any) -> None:
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

    def _aligned_streaming_tensors(
        self,
        predicted: Any,
        observed: Any,
        offset: int,
        *,
        torch: Any,
        dtype: Any,
        device: Any,
    ) -> tuple[Any, Any, int]:
        pred = _as_torch_tensor(predicted, torch=torch, dtype=dtype, device=device).reshape(-1)
        n = min(int(pred.numel()), int(observed.numel()) - offset)
        if n <= 0:
            return pred[:0], observed[:0], offset
        return pred[:n], observed[offset : offset + n], offset + n

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
        output: Any,
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
