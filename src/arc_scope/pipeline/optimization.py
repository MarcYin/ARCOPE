"""Pipeline-level SCOPE parameter optimisation support."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from arc_scope.optim.objective import (
    ScopeObjective,
    _coordinate_aligned_data_arrays,
    _mse_loss,
)
from arc_scope.optim.parameters import (
    ENERGY_BALANCE_OPTIMIZATION_PARAMS,
    SIF_OPTIMIZATION_PARAMS,
    THERMAL_OPTIMIZATION_PARAMS,
    ParameterSet,
    ParameterSpec,
)
from arc_scope.optim.protocols import Optimizer, ScipyOptimizer
from arc_scope.pipeline.config import PipelineConfig

ScopeRunner = Callable[[xr.Dataset], Any]
TorchScopeRunner = Callable[[xr.Dataset, Mapping[str, Any]], Any]


@dataclass
class OptimizationResult:
    """Summary of a completed SCOPE parameter optimisation."""

    status: str
    target_variables: list[str]
    initial_loss: float
    optimized_loss: float
    parameters_initial: dict[str, float]
    parameters_optimized: dict[str, float]
    optimizer: str
    converged: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def optimization_enabled(config: PipelineConfig) -> bool:
    """Return whether the pipeline should run the optimisation branch."""
    if config.optimize:
        return True
    return _as_bool(_normalized_optim_config(config).get("enabled", False))


def run_pipeline_optimization(
    config: PipelineConfig,
    scope_input_ds: xr.Dataset,
    *,
    scope_runner: ScopeRunner | None = None,
) -> tuple[xr.Dataset, xr.Dataset, OptimizationResult]:
    """Optimise configured SCOPE parameters and run the final simulation."""
    optim_config = _normalized_optim_config(config)
    observations = _resolve_observations(optim_config)
    target_variables = _resolve_target_variables(optim_config, observations)
    parameter_set = _resolve_parameter_set(config, optim_config)
    default_autograd_jac = "auto" if scope_runner is not None else "required"
    optimizer, optimizer_name = _build_optimizer(
        optim_config,
        default_autograd_jac=default_autograd_jac,
    )
    runner = scope_runner or _default_scope_runner(config)
    torch_runner = (
        None
        if scope_runner is not None
        or _autograd_disabled_value(
            _resolve_autograd_jac_setting(
                optim_config,
                default=default_autograd_jac,
            )
        )
        else _default_torch_scope_runner(config)
    )
    loss_fn = optim_config.get("loss_fn")
    pixel_selector = optim_config.get("pixel_selector")

    initial_values = _parameter_values(parameter_set)
    objective = ScopeObjective(
        base_dataset=scope_input_ds,
        observations=observations,
        target_variables=target_variables,
        loss_fn=loss_fn,
        scope_runner=runner,
        torch_scope_runner=torch_runner,
        config=config,
        pixel_selector=pixel_selector,
    )

    initial_loss = float(objective.evaluate(initial_values))
    optimized_parameter_set = optimizer.step(objective, parameter_set)
    optimized_values = _parameter_values(optimized_parameter_set)

    optimized_scope_input_ds = optimized_parameter_set.inject_into_dataset(
        scope_input_ds,
        optimized_values,
    )
    optimized_scope_output_ds = runner(optimized_scope_input_ds)
    optimized_loss = _compute_dataset_loss(
        optimized_scope_output_ds,
        observations,
        target_variables,
        loss_fn,
        pixel_selector=pixel_selector,
    )

    optimization_result = OptimizationResult(
        status="optimized",
        target_variables=target_variables,
        initial_loss=initial_loss,
        optimized_loss=optimized_loss,
        parameters_initial=initial_values,
        parameters_optimized=optimized_values,
        optimizer=optimizer_name,
        converged=optimizer.converged(),
        metadata={
            "workflow": config.scope_workflow,
            "n_parameters": len(optimized_parameter_set.specs),
        },
    )
    annotate_optimization_result(optimized_scope_input_ds, optimization_result)
    annotate_optimization_result(optimized_scope_output_ds, optimization_result)
    return optimized_scope_input_ds, optimized_scope_output_ds, optimization_result


def annotate_optimization_result(
    dataset: xr.Dataset,
    result: OptimizationResult,
) -> None:
    """Attach optimization metadata for downstream manifests/exporters."""
    dataset.attrs["arc_scope_optimization_status"] = result.status
    dataset.attrs["arc_scope_optimization_optimizer"] = result.optimizer
    dataset.attrs["arc_scope_optimization_converged"] = str(result.converged).lower()
    dataset.attrs["arc_scope_optimization_targets"] = ",".join(result.target_variables)
    dataset.attrs["arc_scope_optimization_initial_loss"] = float(result.initial_loss)
    dataset.attrs["arc_scope_optimization_optimized_loss"] = float(result.optimized_loss)
    dataset.attrs["arc_scope_optimization_parameters_initial"] = json.dumps(
        result.parameters_initial,
        sort_keys=True,
    )
    dataset.attrs["arc_scope_optimization_parameters_optimized"] = json.dumps(
        result.parameters_optimized,
        sort_keys=True,
    )


def _normalized_optim_config(config: PipelineConfig) -> dict[str, Any]:
    """Support both flat ``optim_config`` and nested ``optim`` payloads."""
    raw = dict(config.optim_config or {})
    nested = raw.get("optim")
    if isinstance(nested, Mapping):
        merged = dict(nested)
        merged.update({key: value for key, value in raw.items() if key != "optim"})
        return merged
    return raw


def _resolve_observations(optim_config: Mapping[str, Any]) -> xr.Dataset:
    """Load observed target data from an optimisation config."""
    observations = None
    for key in ("observations", "observations_ds", "observed"):
        if key in optim_config:
            observations = optim_config[key]
            break
    if isinstance(observations, xr.DataArray):
        name = observations.name or optim_config.get("target_variable") or "target"
        return observations.to_dataset(name=str(name))
    if isinstance(observations, xr.Dataset):
        return observations
    if isinstance(observations, Mapping):
        return xr.Dataset(
            {
                str(name): ("observation", np.asarray(values, dtype=np.float64))
                for name, values in observations.items()
            }
        )

    observations_path = None
    for key in ("observations_path", "observation_path", "observed_path"):
        if key in optim_config:
            observations_path = optim_config[key]
            break
    if observations_path is not None:
        return xr.load_dataset(Path(observations_path))

    raise ValueError(
        "Optimization is enabled, but optim_config does not provide observed "
        "target data. Provide 'observations' as an xarray Dataset/DataArray or "
        "mapping, or provide 'observations_path'."
    )


def _resolve_target_variables(
    optim_config: Mapping[str, Any],
    observations: xr.Dataset,
) -> list[str]:
    """Resolve target variables and verify they exist in the observations."""
    targets = None
    for key in ("target_variables", "targets", "target_variable"):
        if key in optim_config:
            targets = optim_config[key]
            break
    if targets is None:
        target_variables = list(observations.data_vars)
    elif isinstance(targets, str):
        target_variables = [targets]
    else:
        target_variables = [str(target) for target in targets]

    if not target_variables:
        raise ValueError("Optimization requires at least one target variable.")

    missing = [name for name in target_variables if name not in observations]
    if missing:
        raise ValueError(
            "Optimization observations are missing target variables: "
            + ", ".join(missing)
        )
    return target_variables


def _resolve_parameter_set(
    config: PipelineConfig,
    optim_config: Mapping[str, Any],
) -> ParameterSet:
    """Build the ParameterSet requested by optim_config or workflow defaults."""
    parameter_set = optim_config.get("parameter_set")
    if isinstance(parameter_set, ParameterSet):
        return deepcopy(parameter_set)
    if isinstance(parameter_set, str):
        return _preset_parameter_set(parameter_set)

    preset = optim_config.get("parameter_preset") or optim_config.get("preset")
    if preset is not None:
        return _preset_parameter_set(str(preset))

    parameters = (
        optim_config["parameters"]
        if "parameters" in optim_config
        else optim_config.get("params")
    )
    if parameters is not None:
        return _parameter_set_from_config(parameters)

    return _default_parameter_set_for_workflow(config)


def _default_parameter_set_for_workflow(config: PipelineConfig) -> ParameterSet:
    """Choose a conservative parameter preset from the configured workflow."""
    scope_options = config.resolved_scope_options
    if config.scope_workflow == "energy-balance":
        return deepcopy(ENERGY_BALANCE_OPTIMIZATION_PARAMS)
    if config.scope_workflow == "thermal":
        return deepcopy(THERMAL_OPTIMIZATION_PARAMS)
    if config.scope_workflow == "fluorescence" or scope_options.get("calc_fluor"):
        return deepcopy(SIF_OPTIMIZATION_PARAMS)
    raise ValueError(
        "Optimization is enabled, but no default optimisation parameters are "
        f"defined for workflow '{config.scope_workflow}'. Provide "
        "optim_config['parameters'] or optim_config['parameter_set']."
    )


def _preset_parameter_set(name: str) -> ParameterSet:
    normalized = name.lower().replace("_", "-")
    if normalized in {"sif", "fluorescence", "fqe"}:
        return deepcopy(SIF_OPTIMIZATION_PARAMS)
    if normalized == "thermal":
        return deepcopy(THERMAL_OPTIMIZATION_PARAMS)
    if normalized in {"energy-balance", "energy", "full"}:
        return deepcopy(ENERGY_BALANCE_OPTIMIZATION_PARAMS)
    raise ValueError(f"Unknown optimisation parameter preset: {name}")


def _parameter_set_from_config(parameters: Any) -> ParameterSet:
    if isinstance(parameters, ParameterSet):
        return deepcopy(parameters)
    if isinstance(parameters, Mapping):
        specs = [
            _parameter_spec_from_mapping({"name": name, **dict(spec)})
            for name, spec in parameters.items()
        ]
        return ParameterSet(specs)
    if isinstance(parameters, Sequence) and not isinstance(parameters, (str, bytes)):
        specs = []
        for spec in parameters:
            if isinstance(spec, ParameterSpec):
                specs.append(deepcopy(spec))
            elif isinstance(spec, Mapping):
                specs.append(_parameter_spec_from_mapping(spec))
            else:
                raise TypeError("Each optimisation parameter must be a mapping.")
        return ParameterSet(specs)
    raise TypeError("optim_config['parameters'] must be a mapping or sequence.")


def _parameter_spec_from_mapping(spec: Mapping[str, Any]) -> ParameterSpec:
    required = ("name", "initial", "lower", "upper")
    missing = [name for name in required if name not in spec]
    if missing:
        raise ValueError(
            "Optimisation parameter spec is missing required fields: "
            + ", ".join(missing)
        )
    return ParameterSpec(
        name=str(spec["name"]),
        initial=float(spec["initial"]),
        lower=float(spec["lower"]),
        upper=float(spec["upper"]),
        optimize=_as_bool(spec.get("optimize", True)),
        transform=str(spec.get("transform", "identity")),
    )


def _build_optimizer(
    optim_config: Mapping[str, Any],
    *,
    default_autograd_jac: bool | str = "required",
) -> tuple[Optimizer, str]:
    """Build the configured optimiser."""
    optimizer_config, optimizer_type, optimizer_options = _optimizer_config_parts(optim_config)
    if isinstance(optimizer_config, Optimizer):
        return optimizer_config, optimizer_config.__class__.__name__

    if optimizer_type.lower() not in {"scipy", "scipy-minimize", "minimize"}:
        raise ValueError(
            "Pipeline optimization currently supports only the 'scipy' optimizer."
        )

    method = str(
        optim_config.get("method")
        or optimizer_options.get("method")
        or "L-BFGS-B"
    )
    max_iter = int(
        optim_config.get("max_iter")
        or optim_config.get("max_iterations")
        or optimizer_options.get("max_iter")
        or optimizer_options.get("max_iterations")
        or 100
    )
    tol = float(optim_config.get("tol") or optimizer_options.get("tol") or 1e-6)
    use_autograd_jac = _resolve_autograd_jac_setting(
        optim_config,
        default=default_autograd_jac,
    )
    return (
        ScipyOptimizer(
            method=method,
            max_iter=max_iter,
            tol=tol,
            use_autograd_jac=use_autograd_jac,
        ),
        f"scipy:{method}",
    )


def _optimizer_config_parts(
    optim_config: Mapping[str, Any],
) -> tuple[Any, str, dict[str, Any]]:
    optimizer_config = optim_config.get("optimizer", "scipy")
    if isinstance(optimizer_config, Optimizer):
        return optimizer_config, optimizer_config.__class__.__name__, {}
    if isinstance(optimizer_config, Mapping):
        optimizer_type = str(
            optimizer_config.get("type", optimizer_config.get("name", "scipy"))
        )
        return optimizer_config, optimizer_type, dict(optimizer_config)
    return optimizer_config, str(optimizer_config), {}


def _resolve_autograd_jac_setting(
    optim_config: Mapping[str, Any],
    *,
    default: bool | str,
) -> bool | str:
    _, _, optimizer_options = _optimizer_config_parts(optim_config)
    return (
        optim_config.get("use_autograd_jac")
        if "use_autograd_jac" in optim_config
        else optim_config.get(
            "autograd_jac",
            optimizer_options.get(
                "use_autograd_jac",
                optimizer_options.get("autograd_jac", default),
            ),
        )
    )


def _autograd_disabled_value(value: Any) -> bool:
    if value is False:
        return True
    return str(value).lower() in {
        "0",
        "false",
        "no",
        "off",
        "disable",
        "disabled",
        "none",
    }


def _parameter_values(parameter_set: ParameterSet) -> dict[str, float]:
    return {spec.name: float(spec.initial) for spec in parameter_set.specs}


def _compute_dataset_loss(
    output: xr.Dataset,
    observations: xr.Dataset,
    target_variables: Sequence[str],
    loss_fn: Any,
    *,
    pixel_selector: Mapping[str, Any] | None = None,
) -> float:
    """Compute the configured scalar loss for a SCOPE output dataset."""
    loss = loss_fn or _mse_loss
    total_loss = 0.0
    compared = False
    for variable in target_variables:
        if variable not in output:
            raise ValueError(f"SCOPE output is missing target variable '{variable}'.")
        if variable not in observations:
            raise ValueError(
                f"Optimization observations are missing target variable '{variable}'."
            )
        predicted_da, observed_da = _coordinate_aligned_data_arrays(
            output[variable],
            observations[variable],
            var_name=variable,
            pixel_selector=pixel_selector,
        )
        predicted = np.asarray(predicted_da.values).reshape(-1)
        observed = np.asarray(observed_da.values).reshape(-1)
        mask = np.isfinite(predicted) & np.isfinite(observed)
        if mask.any():
            compared = True
            total_loss += float(loss(predicted[mask], observed[mask]))

    if not compared:
        raise ValueError("Optimization loss has no finite prediction/observation pairs.")
    return float(total_loss)


def _default_scope_runner(config: PipelineConfig) -> ScopeRunner:
    def _run(dataset: xr.Dataset) -> xr.Dataset:
        from arc_scope.pipeline.steps import run_scope_simulation

        return run_scope_simulation(dataset, config)

    return _run


def _default_torch_scope_runner(config: PipelineConfig) -> TorchScopeRunner:
    return _DefaultTorchScopeRunner(config)


class _DefaultTorchScopeRunner:
    """Default differentiable SCOPE runner with chunk streaming support."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def __call__(
        self,
        dataset: xr.Dataset,
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        from arc_scope.pipeline.steps import run_scope_simulation_tensors

        return run_scope_simulation_tensors(
            dataset,
            self._config,
            parameter_values=params,
        )

    def iter_chunks(
        self,
        dataset: xr.Dataset,
        params: Mapping[str, Any],
    ):
        from arc_scope.pipeline.steps import iter_scope_simulation_tensor_chunks

        yield from iter_scope_simulation_tensor_chunks(
            dataset,
            self._config,
            parameter_values=params,
        )


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
