"""SCOPE forward pass wrapped as a differentiable objective function.

Wraps the SCOPE simulation pipeline so it can be used as an optimisation
target.  The objective injects parameters into the prepared dataset, runs
SCOPE, and computes a scalar loss against observations.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
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
    pixel_selector:
        Optional selector for prediction dimensions that are not present in
        the observations, for example ``{"y": y_coord, "x": x_coord}``.
        Use ``{"y": {"isel": y_index}, "x": {"isel": x_index}}`` for
        positional selection. Extra prediction dimensions must be selected
        explicitly rather than silently flattened.
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
        pixel_selector: Mapping[str, Any] | None = None,
    ):
        self._base_dataset = base_dataset
        self._observations = observations
        self._target_variables = list(target_variables)
        self._loss_fn = loss_fn or _mse_loss
        self._scope_runner = scope_runner
        self._torch_scope_runner = torch_scope_runner
        self._config = config
        self._pixel_selector = dict(pixel_selector) if pixel_selector is not None else None

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
            pred_da, obs_da = _coordinate_aligned_data_arrays(
                output[var],
                self._observations[var],
                var_name=var,
                pixel_selector=self._pixel_selector,
            )
            pred = np.asarray(pred_da.values).reshape(-1)
            obs = np.asarray(obs_da.values).reshape(-1)
            mask = np.isfinite(pred) & np.isfinite(obs)
            if mask.any():
                compared = True
                total_loss += float(self._loss_fn(pred[mask], obs[mask]))

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
            var: self._observations[var]
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
        observation_tensors: dict[str, xr.DataArray],
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
        observation_tensors: dict[str, xr.DataArray],
        counts: dict[str, int],
        *,
        torch: Any,
        dtype: Any,
        device: Any,
    ) -> None:
        # Accumulate chunk gradients at the physical-parameter boundary, then
        # apply the parameter transform chain rule once at the end.
        grad_inputs = [
            (name, value)
            for name, value in params.items()
            if _is_torch_tensor(value)
            and bool(getattr(value, "requires_grad", False))
        ]
        if not grad_inputs:
            raise AutogradUnavailable(
                "Streaming autograd loss is not connected to any optimisable "
                "physical parameters."
            )
        grad_totals = {
            name: torch.zeros_like(value)
            for name, value in grad_inputs
        }
        saw_gradient = False
        offsets = {var: 0 for var in self._target_variables}
        outputs = iter(self._iter_scope_torch_chunks(self._base_dataset, params))
        try:
            output = next(outputs)
        except StopIteration:
            output = None

        while output is not None:
            # Some custom runners yield slices from a shared graph. Keep that
            # graph only until the final yielded chunk, then release it.
            try:
                next_output = next(outputs)
                retain_graph = True
            except StopIteration:
                next_output = None
                retain_graph = False

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
                if not bool(getattr(chunk_loss, "requires_grad", False)):
                    raise AutogradUnavailable(
                        "Streaming chunk loss is detached from the optimisable "
                        "parameters."
                    )
                grads = torch.autograd.grad(
                    chunk_loss,
                    tuple(value for _, value in grad_inputs),
                    allow_unused=True,
                    retain_graph=retain_graph,
                )
                for (name, _), grad in zip(grad_inputs, grads):
                    if grad is not None:
                        grad_totals[name] = grad_totals[name] + grad.detach()
                        saw_gradient = True
            del output
            output = next_output

        if not saw_gradient:
            raise AutogradUnavailable(
                "Streaming autograd did not produce gradients for optimisable "
                "parameters."
            )
        torch.autograd.backward(
            tuple(value for _, value in grad_inputs),
            grad_tensors=tuple(grad_totals[name] for name, _ in grad_inputs),
        )

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
        observed: xr.DataArray,
        offset: int,
        *,
        torch: Any,
        dtype: Any,
        device: Any,
    ) -> tuple[Any, Any, int]:
        return self._aligned_torch_tensors(
            predicted,
            observed,
            offset=offset,
            torch=torch,
            dtype=dtype,
            device=device,
            allow_empty=True,
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
            pred, obs, _ = self._aligned_torch_tensors(
                output[var],
                self._observations[var],
                offset=None,
                torch=torch,
                dtype=dtype,
                device=device,
                allow_empty=False,
            )
            mask = torch.isfinite(pred) & torch.isfinite(obs)
            if bool(mask.any().detach().cpu().item()):
                compared = True
                pred_valid = pred[mask]
                obs_valid = obs[mask]
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

    def _aligned_torch_tensors(
        self,
        predicted: Any,
        observed: xr.DataArray,
        *,
        offset: int | None,
        torch: Any,
        dtype: Any,
        device: Any,
        allow_empty: bool,
    ) -> tuple[Any, Any, int]:
        (
            pred_tensor,
            pred_dims,
            pred_shape,
            pred_coords,
            pred_indexed,
            pred_positions,
            next_offset,
        ) = self._torch_prediction_layout(
            predicted,
            observed,
            offset=offset,
            torch=torch,
            dtype=dtype,
            device=device,
        )
        pred_indices, pred_values = _position_values(
            pred_dims,
            pred_shape,
            pred_coords,
            pred_positions,
        )
        pred, obs = _coordinate_aligned_flat_torch_tensors(
            pred_tensor,
            pred_dims=pred_dims,
            pred_shape=pred_shape,
            pred_coords=pred_coords,
            pred_indexed=pred_indexed,
            pred_indices=pred_indices,
            pred_values=pred_values,
            observed=observed,
            var_name=observed.name or "target",
            pixel_selector=self._pixel_selector,
            torch=torch,
            dtype=dtype,
            device=device,
            allow_empty=allow_empty,
        )
        return pred, obs, next_offset

    def _torch_prediction_layout(
        self,
        predicted: Any,
        observed: xr.DataArray,
        *,
        offset: int | None,
        torch: Any,
        dtype: Any,
        device: Any,
    ) -> tuple[Any, tuple[str, ...], tuple[int, ...], dict[str, np.ndarray], dict[str, bool], np.ndarray, int]:
        if isinstance(predicted, xr.DataArray):
            tensor = _as_torch_tensor(predicted, torch=torch, dtype=dtype, device=device).reshape(-1)
            dims = tuple(str(dim) for dim in predicted.dims)
            shape = tuple(int(predicted.sizes[dim]) for dim in dims)
            coords, indexed = _dimension_coordinates(predicted, dims, shape)
            positions = np.arange(int(np.prod(shape, dtype=np.int64)), dtype=np.int64)
            next_offset = (offset or 0) + int(tensor.numel()) if offset is not None else 0
            return tensor, dims, shape, coords, indexed, positions, next_offset

        raw = _as_torch_tensor(predicted, torch=torch, dtype=dtype, device=device)
        tensor = raw.reshape(-1)
        count = int(tensor.numel())
        if offset is None:
            dims, shape, coord_source = self._infer_full_prediction_layout(raw, observed)
            positions = np.arange(count, dtype=np.int64)
            next_offset = 0
        else:
            dims, shape, coord_source = self._infer_streaming_prediction_layout(
                count,
                observed,
                offset=offset,
            )
            positions = np.arange(offset, offset + count, dtype=np.int64)
            next_offset = offset + count

        source = self._base_dataset if coord_source == "base" else observed
        coords, indexed = _dimension_coordinates(source, dims, shape)
        return tensor, dims, shape, coords, indexed, positions, next_offset

    def _infer_full_prediction_layout(
        self,
        tensor: Any,
        observed: xr.DataArray,
    ) -> tuple[tuple[str, ...], tuple[int, ...], str]:
        tensor_shape = tuple(int(size) for size in getattr(tensor, "shape", ()))
        tensor_count = int(tensor.numel())
        base_dims = self._base_prediction_dims()
        base_shape = tuple(int(self._base_dataset.sizes[dim]) for dim in base_dims)
        observed_dims = tuple(str(dim) for dim in observed.dims)
        observed_shape = tuple(int(observed.sizes[dim]) for dim in observed_dims)

        if tensor_count == 1 and not base_dims and not observed_dims:
            return (), (), "observed"
        if base_dims and (
            tensor_shape == base_shape or tensor_count == _shape_size(base_shape)
        ):
            return base_dims, base_shape, "base"
        if tensor_shape == observed_shape or tensor_count == int(observed.size):
            return observed_dims, observed_shape, "observed"
        raise ValueError(
            "Prediction tensor shape cannot be aligned to observations by "
            "coordinates. Return an xarray.DataArray with coordinates or use "
            "a tensor shape compatible with the objective base dataset."
        )

    def _infer_streaming_prediction_layout(
        self,
        count: int,
        observed: xr.DataArray,
        *,
        offset: int,
    ) -> tuple[tuple[str, ...], tuple[int, ...], str]:
        base_dims = self._base_prediction_dims()
        base_shape = tuple(int(self._base_dataset.sizes[dim]) for dim in base_dims)
        if base_dims and offset + count <= _shape_size(base_shape):
            return base_dims, base_shape, "base"

        observed_dims = tuple(str(dim) for dim in observed.dims)
        observed_shape = tuple(int(observed.sizes[dim]) for dim in observed_dims)
        if offset + count <= int(observed.size):
            return observed_dims, observed_shape, "observed"

        raise ValueError(
            "Streaming prediction chunk cannot be aligned to observations by "
            "coordinates. Return xarray.DataArray chunks with coordinates."
        )

    def _base_prediction_dims(self) -> tuple[str, ...]:
        preferred = tuple(
            dim for dim in ("y", "x", "time") if dim in self._base_dataset.sizes
        )
        if preferred:
            return preferred
        return tuple(str(dim) for dim in self._base_dataset.sizes)


def _coordinate_aligned_data_arrays(
    predicted: xr.DataArray,
    observed: xr.DataArray,
    *,
    var_name: str,
    pixel_selector: Mapping[str, Any] | None,
) -> tuple[xr.DataArray, xr.DataArray]:
    pred_da = _ensure_dataarray(predicted, name=var_name)
    obs_da = _ensure_dataarray(observed, name=var_name)
    pred_da = _select_extra_prediction_dims(
        pred_da,
        obs_da,
        var_name=var_name,
        pixel_selector=pixel_selector,
    )

    extra_obs_dims = sorted(set(obs_da.dims) - set(pred_da.dims))
    if extra_obs_dims:
        raise ValueError(
            f"Observations for {var_name!r} have dims {extra_obs_dims} "
            "that are not present in the prediction."
        )

    try:
        pred_da, obs_da = xr.align(pred_da, obs_da, join="inner")
    except ValueError as exc:
        raise ValueError(
            f"Prediction and observation for {var_name!r} cannot be aligned "
            "by coordinates."
        ) from exc

    if pred_da.size == 0 or obs_da.size == 0:
        raise ValueError(
            f"Prediction and observation for {var_name!r} have no "
            "overlapping coordinates."
        )
    for dim in obs_da.dims:
        if int(obs_da.sizes[dim]) == 0:
            raise ValueError(
                f"Prediction and observation for {var_name!r} have no "
                f"overlapping coordinates on dimension {dim!r}."
            )
    if pred_da.dims != obs_da.dims:
        pred_da = pred_da.transpose(*obs_da.dims)
    return pred_da, obs_da


def _ensure_dataarray(value: Any, *, name: str) -> xr.DataArray:
    if isinstance(value, xr.DataArray):
        return value
    return xr.DataArray(np.asarray(value), name=name)


def _select_extra_prediction_dims(
    predicted: xr.DataArray,
    observed: xr.DataArray,
    *,
    var_name: str,
    pixel_selector: Mapping[str, Any] | None,
) -> xr.DataArray:
    extra_dims = [dim for dim in predicted.dims if dim not in observed.dims]
    if not extra_dims:
        return predicted
    if pixel_selector is None:
        raise ValueError(
            f"Prediction for {var_name!r} has extra dims {sorted(extra_dims)} "
            "but no pixel_selector was provided."
        )

    missing = [dim for dim in extra_dims if dim not in pixel_selector]
    if missing:
        raise ValueError(
            f"Prediction for {var_name!r} has extra dims {sorted(extra_dims)} "
            f"but pixel_selector is missing {sorted(missing)}."
        )

    sel_kwargs: dict[str, Any] = {}
    isel_kwargs: dict[str, Any] = {}
    for dim in extra_dims:
        value = pixel_selector[dim]
        if _is_index_selector(value):
            isel_kwargs[dim] = _index_selector_value(value)
        else:
            sel_kwargs[dim] = value

    try:
        if sel_kwargs:
            predicted = predicted.sel(sel_kwargs)
        if isel_kwargs:
            predicted = predicted.isel(isel_kwargs)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"pixel_selector could not select prediction for {var_name!r}."
        ) from exc

    remaining = [dim for dim in predicted.dims if dim not in observed.dims]
    if remaining:
        raise ValueError(
            f"pixel_selector for {var_name!r} must select scalar values for "
            f"extra dims {sorted(remaining)}."
        )
    return predicted


def _coordinate_aligned_flat_torch_tensors(
    pred_tensor: Any,
    *,
    pred_dims: tuple[str, ...],
    pred_shape: tuple[int, ...],
    pred_coords: dict[str, np.ndarray],
    pred_indexed: dict[str, bool],
    pred_indices: dict[str, np.ndarray],
    pred_values: dict[str, np.ndarray],
    observed: xr.DataArray,
    var_name: str,
    pixel_selector: Mapping[str, Any] | None,
    torch: Any,
    dtype: Any,
    device: Any,
    allow_empty: bool,
) -> tuple[Any, Any]:
    _ = pred_shape
    obs_dims = tuple(str(dim) for dim in observed.dims)
    extra_pred_dims = [dim for dim in pred_dims if dim not in obs_dims]
    selected = np.ones(int(pred_tensor.numel()), dtype=bool)

    if extra_pred_dims:
        if pixel_selector is None:
            raise ValueError(
                f"Prediction for {var_name!r} has extra dims "
                f"{sorted(extra_pred_dims)} but no pixel_selector was provided."
            )
        missing = [dim for dim in extra_pred_dims if dim not in pixel_selector]
        if missing:
            raise ValueError(
                f"Prediction for {var_name!r} has extra dims "
                f"{sorted(extra_pred_dims)} but pixel_selector is missing "
                f"{sorted(missing)}."
            )
        for dim in extra_pred_dims:
            selected &= _selector_mask(
                pred_values[dim],
                pred_indices[dim],
                pixel_selector[dim],
            )

    extra_obs_dims = sorted(set(obs_dims) - set(pred_dims))
    if extra_obs_dims:
        raise ValueError(
            f"Observations for {var_name!r} have dims {extra_obs_dims} "
            "that are not present in the prediction."
        )

    remaining_pred_dims = [dim for dim in pred_dims if dim not in extra_pred_dims]
    if set(remaining_pred_dims) != set(obs_dims):
        raise ValueError(
            f"Prediction and observation for {var_name!r} cannot be aligned "
            "by coordinates."
        )

    if not bool(selected.any()):
        if allow_empty:
            return _empty_torch(pred_tensor), _empty_torch(pred_tensor)
        raise ValueError(
            f"pixel_selector for {var_name!r} did not match any prediction "
            "coordinates."
        )

    obs_coords, obs_indexed = _dimension_coordinates(
        observed,
        obs_dims,
        tuple(int(observed.sizes[dim]) for dim in obs_dims),
    )
    obs_indices, obs_values = _position_values(
        obs_dims,
        tuple(int(observed.sizes[dim]) for dim in obs_dims),
        obs_coords,
        np.arange(int(observed.size), dtype=np.int64),
    )

    for dim in obs_dims:
        if not (pred_indexed.get(dim, False) and obs_indexed.get(dim, False)):
            pred_size = len(pred_coords[dim])
            obs_size = len(obs_coords[dim])
            if pred_size != obs_size:
                raise ValueError(
                    f"Prediction and observation for {var_name!r} cannot be "
                    f"positionally aligned on unindexed dimension {dim!r}: "
                    f"{pred_size} != {obs_size}."
                )

    pred_take_all = np.flatnonzero(selected)
    pred_key_arrays = []
    obs_key_arrays = []
    for dim in obs_dims:
        use_labels = pred_indexed.get(dim, False) and obs_indexed.get(dim, False)
        pred_key_arrays.append(
            pred_values[dim][selected] if use_labels else pred_indices[dim][selected]
        )
        obs_key_arrays.append(obs_values[dim] if use_labels else obs_indices[dim])

    pred_keys = _coordinate_index(pred_key_arrays, obs_dims)
    obs_keys = _coordinate_index(obs_key_arrays, obs_dims)
    if not pred_keys.is_unique or not obs_keys.is_unique:
        raise ValueError(
            f"Prediction and observation for {var_name!r} must have unique "
            "coordinate keys for objective alignment."
        )

    matches = pred_keys.isin(obs_keys)
    if not bool(np.any(matches)):
        if allow_empty:
            return _empty_torch(pred_tensor), _empty_torch(pred_tensor)
        raise ValueError(
            f"Prediction and observation for {var_name!r} have no "
            "overlapping coordinates."
        )

    pred_take = pred_take_all[matches]
    obs_take = obs_keys.get_indexer(pred_keys[matches])
    pred_selected = _take_torch(pred_tensor.reshape(-1), pred_take, torch, device)
    obs_tensor = _as_torch_tensor(observed, torch=torch, dtype=dtype, device=device)
    obs_selected = _take_torch(obs_tensor.reshape(-1), obs_take, torch, device)
    return pred_selected, obs_selected


def _position_values(
    dims: tuple[str, ...],
    shape: tuple[int, ...],
    coords: dict[str, np.ndarray],
    positions: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not dims:
        empty = np.zeros(len(positions), dtype=np.int64)
        return {}, {"__scalar__": empty}

    if len(positions) == 0:
        indices = tuple(np.array([], dtype=np.int64) for _ in dims)
    else:
        indices = np.unravel_index(positions, shape, order="C")
    index_values = {
        dim: np.asarray(indices[i], dtype=np.int64)
        for i, dim in enumerate(dims)
    }
    coord_values = {
        dim: np.asarray(coords[dim])[index_values[dim]]
        for dim in dims
    }
    return index_values, coord_values


def _dimension_coordinates(
    source: xr.Dataset | xr.DataArray,
    dims: tuple[str, ...],
    shape: tuple[int, ...],
) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    coords: dict[str, np.ndarray] = {}
    indexed: dict[str, bool] = {}
    for dim, size in zip(dims, shape):
        coord = source.coords.get(dim)
        if coord is not None and tuple(coord.dims) == (dim,) and int(coord.size) == size:
            coords[dim] = np.asarray(coord.values)
            indexed[dim] = True
        else:
            coords[dim] = np.arange(size, dtype=np.int64)
            indexed[dim] = False
    return coords, indexed


def _coordinate_index(arrays: list[np.ndarray], names: tuple[str, ...]) -> pd.Index:
    if not arrays:
        return pd.Index([0], name="__scalar__")
    if len(arrays) == 1:
        return pd.Index(arrays[0], name=names[0])
    return pd.MultiIndex.from_arrays(arrays, names=names)


def _selector_mask(
    coord_values: np.ndarray,
    index_values: np.ndarray,
    selector: Any,
) -> np.ndarray:
    if _is_index_selector(selector):
        return index_values == int(_index_selector_value(selector))
    return coord_values == selector


def _is_index_selector(value: Any) -> bool:
    return isinstance(value, Mapping) and any(
        key in value for key in ("isel", "index", "position")
    )


def _index_selector_value(value: Mapping[str, Any]) -> Any:
    for key in ("isel", "index", "position"):
        if key in value:
            return value[key]
    raise ValueError("Index selector must contain 'isel', 'index', or 'position'.")


def _shape_size(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    return int(np.prod(shape, dtype=np.int64))


def _take_torch(tensor: Any, indices: np.ndarray, torch: Any, device: Any) -> Any:
    if len(indices) == 0:
        return _empty_torch(tensor)
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
    return tensor.index_select(0, index_tensor)


def _empty_torch(tensor: Any) -> Any:
    return tensor.reshape(-1)[:0]


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
