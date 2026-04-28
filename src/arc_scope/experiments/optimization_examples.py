"""Generate reproducible optimisation example artifacts for the docs.

The examples in this module exercise ARC-SCOPE's real optimisation machinery
with lightweight deterministic proxy runners. They are intended for
documentation and regression-style inspection when ARC and scope-rtm are not
available. They do not claim to be validated SCOPE scientific products.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from arc_scope.data import TEST_FIELD_GEOJSON
from arc_scope.pipeline.config import PipelineConfig
from arc_scope.pipeline.optimization import OptimizationResult, run_pipeline_optimization


DEFAULT_OUTPUT_DIR = Path("docs/assets/optimization")


def generate_optimization_examples(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Run the proxy optimisation examples and write docs artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cases = [
        _run_sif_example(),
        _run_thermal_example(),
        _run_energy_balance_example(),
    ]

    timeseries = pd.concat([case["timeseries"] for case in cases], ignore_index=True)
    parameters = pd.concat([case["parameters"] for case in cases], ignore_index=True)
    summary = {
        case["key"]: {
            "title": case["title"],
            "workflow": case["workflow"],
            "target_variables": case["result"].target_variables,
            "initial_loss": case["result"].initial_loss,
            "optimized_loss": case["result"].optimized_loss,
            "converged": case["result"].converged,
            "parameters_initial": case["result"].parameters_initial,
            "parameters_optimized": case["result"].parameters_optimized,
            "parameters_true": case["true_parameters"],
            "optimizer": case["result"].optimizer,
        }
        for case in cases
    }

    files = {
        "summary": output_path / "summary.json",
        "timeseries": output_path / "timeseries.csv",
        "parameters": output_path / "parameter_summary.csv",
        "sif_fit": output_path / "sif_fit.svg",
        "thermal_fit": output_path / "thermal_fit.svg",
        "energy_balance_fit": output_path / "energy_balance_fit.svg",
        "parameter_summary": output_path / "parameter_summary.svg",
    }

    files["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    timeseries.to_csv(files["timeseries"], index=False)
    parameters.to_csv(files["parameters"], index=False)

    _plot_single_target(
        cases[0],
        target="F740",
        ylabel="Proxy SIF F740",
        output_path=files["sif_fit"],
    )
    _plot_single_target(
        cases[1],
        target="Loutt",
        ylabel="Proxy thermal radiance Loutt",
        output_path=files["thermal_fit"],
    )
    _plot_energy_balance(cases[2], files["energy_balance_fit"])
    _plot_parameter_summary(parameters, files["parameter_summary"])

    return files


def _run_sif_example() -> dict[str, Any]:
    time = _example_time()
    base_ds = _base_dataset(time)
    true_parameters = {"fqe": 0.018}
    observations = _add_noise(
        _sif_proxy_runner(_inject_values(base_ds, true_parameters)),
        {"F740": 0.010},
    )
    config = _example_config(
        workflow="fluorescence",
        optim_config={
            "enabled": True,
            "observations": observations[["F740"]],
            "target_variables": ["F740"],
            "parameters": [
                {
                    "name": "fqe",
                    "initial": 0.01,
                    "lower": 0.001,
                    "upper": 0.1,
                    "transform": "log",
                }
            ],
            "optimizer": {"type": "scipy", "method": "L-BFGS-B"},
            "max_iter": 80,
            "tol": 1e-10,
        },
    )
    return _run_case(
        key="sif",
        title="SIF fit",
        workflow="fluorescence",
        config=config,
        base_ds=base_ds,
        observations=observations[["F740"]],
        true_parameters=true_parameters,
        runner=_sif_proxy_runner,
    )


def _run_thermal_example() -> dict[str, Any]:
    time = _example_time()
    base_ds = _base_dataset(time)
    true_parameters = {"Tcu": 29.5, "Tch": 26.5, "Tsu": 34.0, "Tsh": 29.0}
    observations = _add_noise(
        _thermal_proxy_runner(_inject_values(base_ds, true_parameters)),
        {"Loutt": 0.030, "Eoutt": 0.026},
    )
    config = _example_config(
        workflow="thermal",
        optim_config={
            "enabled": True,
            "observations": observations[["Loutt", "Eoutt"]],
            "target_variables": ["Loutt", "Eoutt"],
            "parameter_preset": "thermal",
            "optimizer": "scipy",
            "max_iter": 100,
            "tol": 1e-10,
        },
    )
    return _run_case(
        key="thermal",
        title="Thermal fit",
        workflow="thermal",
        config=config,
        base_ds=base_ds,
        observations=observations[["Loutt", "Eoutt"]],
        true_parameters=true_parameters,
        runner=_thermal_proxy_runner,
    )


def _run_energy_balance_example() -> dict[str, Any]:
    time = _example_time()
    base_ds = _base_dataset(time)
    true_parameters = {
        "fqe": 0.016,
        "rss": 720.0,
        "rbs": 14.0,
        "Cd": 0.28,
        "rwc": 0.62,
    }
    observations = _add_noise(
        _energy_balance_proxy_runner(_inject_values(base_ds, true_parameters)),
        {"F740": 0.012, "Loutt": 0.025, "LE": 0.090},
    )
    config = _example_config(
        workflow="energy-balance",
        optim_config={
            "enabled": True,
            "observations": observations[["F740", "Loutt", "LE"]],
            "target_variables": ["F740", "Loutt", "LE"],
            "parameter_preset": "energy-balance",
            "optimizer": {"type": "scipy", "method": "L-BFGS-B"},
            "max_iter": 160,
            "tol": 1e-10,
        },
    )
    return _run_case(
        key="energy_balance",
        title="Coupled energy-balance fit",
        workflow="energy-balance",
        config=config,
        base_ds=base_ds,
        observations=observations[["F740", "Loutt", "LE"]],
        true_parameters=true_parameters,
        runner=_energy_balance_proxy_runner,
    )


def _run_case(
    *,
    key: str,
    title: str,
    workflow: str,
    config: PipelineConfig,
    base_ds: xr.Dataset,
    observations: xr.Dataset,
    true_parameters: Mapping[str, float],
    runner: Callable[[xr.Dataset], xr.Dataset],
) -> dict[str, Any]:
    optimized_input, optimized_output, result = run_pipeline_optimization(
        config,
        base_ds,
        scope_runner=runner,
    )
    initial_output = runner(_inject_values(base_ds, result.parameters_initial))

    timeseries = _timeseries_frame(
        key=key,
        observations=observations,
        initial_output=initial_output,
        optimized_output=optimized_output,
    )
    parameters = _parameter_frame(
        key=key,
        result=result,
        true_parameters=true_parameters,
    )
    return {
        "key": key,
        "title": title,
        "workflow": workflow,
        "result": result,
        "base_ds": base_ds,
        "optimized_input": optimized_input,
        "observations": observations,
        "initial_output": initial_output,
        "optimized_output": optimized_output,
        "timeseries": timeseries,
        "parameters": parameters,
        "true_parameters": dict(true_parameters),
    }


def _example_config(workflow: str, optim_config: dict[str, Any]) -> PipelineConfig:
    return PipelineConfig(
        geojson_path=TEST_FIELD_GEOJSON,
        start_date="2021-05-15",
        end_date="2021-10-01",
        crop_type="wheat",
        start_of_season=170,
        year=2021,
        scope_workflow=workflow,
        save_arc_npz=False,
        save_scope_netcdf=False,
        optim_config=optim_config,
    )


def _example_time() -> pd.DatetimeIndex:
    return pd.date_range("2021-06-01", periods=18, freq="7D")


def _base_dataset(time: Sequence[pd.Timestamp]) -> xr.Dataset:
    n = len(time)
    phase = np.linspace(0.0, 1.0, n)
    return xr.Dataset(
        {
            "apar": ("time", 420.0 + 120.0 * np.sin(np.pi * phase)),
            "water_modifier": ("time", 0.82 + 0.12 * np.cos(2.0 * np.pi * phase)),
            "thermal_base": ("time", 405.0 + 5.0 * np.sin(1.4 * np.pi * phase)),
            "soil_sensitivity": ("time", 0.0034 + 0.0010 * phase),
            "boundary_sensitivity": ("time", 0.065 + 0.025 * np.cos(np.pi * phase)),
            "le_base": ("time", 82.0 + 12.0 * np.sin(np.pi * phase)),
        },
        coords={"time": pd.DatetimeIndex(time)},
    )


def _sif_proxy_runner(dataset: xr.Dataset) -> xr.Dataset:
    fqe = _scalar(dataset, "fqe", 0.01)
    f740 = fqe * dataset["apar"] * dataset["water_modifier"]
    f685 = 0.82 * fqe * dataset["apar"] * (1.0 + 0.05 * dataset["water_modifier"])
    return xr.Dataset({"F740": f740, "F685": f685}, coords={"time": dataset.coords["time"]})


def _thermal_proxy_runner(dataset: xr.Dataset) -> xr.Dataset:
    tcu = _scalar(dataset, "Tcu", 25.0)
    tch = _scalar(dataset, "Tch", 24.0)
    tsu = _scalar(dataset, "Tsu", 30.0)
    tsh = _scalar(dataset, "Tsh", 27.0)
    loutt = (
        dataset["thermal_base"]
        + dataset["boundary_sensitivity"] * (tcu - 25.0)
        + 0.20 * (tch - 24.0)
        + dataset["soil_sensitivity"] * (tsu - 30.0)
        + 0.35 * (tsh - 27.0)
    )
    eoutt = (
        0.97 * dataset["thermal_base"]
        + 0.15 * (tcu - 25.0)
        + dataset["boundary_sensitivity"] * (tch - 24.0)
        + 0.25 * (tsu - 30.0)
        + dataset["soil_sensitivity"] * (tsh - 27.0)
        + 2.5
    )
    return xr.Dataset({"Loutt": loutt, "Eoutt": eoutt}, coords={"time": dataset.coords["time"]})


def _energy_balance_proxy_runner(dataset: xr.Dataset) -> xr.Dataset:
    fqe = _scalar(dataset, "fqe", 0.01)
    rss = _scalar(dataset, "rss", 500.0)
    rbs = _scalar(dataset, "rbs", 10.0)
    cd = _scalar(dataset, "Cd", 0.2)
    rwc = _scalar(dataset, "rwc", 0.5)
    f740 = fqe * dataset["apar"] * (0.70 + 0.45 * rwc)
    loutt = (
        dataset["thermal_base"]
        + dataset["soil_sensitivity"] * (rss - 500.0)
        + dataset["boundary_sensitivity"] * (rbs - 10.0)
        - 0.65 * (rwc - 0.5)
    )
    le = dataset["le_base"] * (0.72 + 0.50 * rwc) + 32.0 * (cd - 0.2) - 0.004 * (rss - 500.0)
    return xr.Dataset(
        {"F740": f740, "Loutt": loutt, "LE": le},
        coords={"time": dataset.coords["time"]},
    )


def _inject_values(dataset: xr.Dataset, values: Mapping[str, float]) -> xr.Dataset:
    updated = dataset.copy(deep=True)
    for name, value in values.items():
        updated[name] = float(value)
    return updated


def _scalar(dataset: xr.Dataset, name: str, default: float) -> float:
    if name not in dataset:
        return default
    return float(np.asarray(dataset[name]).ravel()[0])


def _add_noise(dataset: xr.Dataset, amplitudes: Mapping[str, float]) -> xr.Dataset:
    phase = np.linspace(0.0, 2.0 * np.pi, dataset.sizes["time"])
    noisy = dataset.copy(deep=True)
    for index, (name, amplitude) in enumerate(amplitudes.items()):
        if name in noisy:
            perturbation = amplitude * np.sin(phase + index * np.pi / 5.0)
            noisy[name] = noisy[name] + xr.DataArray(perturbation, coords={"time": noisy.coords["time"]})
    return noisy


def _timeseries_frame(
    *,
    key: str,
    observations: xr.Dataset,
    initial_output: xr.Dataset,
    optimized_output: xr.Dataset,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in observations.data_vars:
        time_values = pd.to_datetime(observations.coords["time"].values)
        for i, timestamp in enumerate(time_values):
            rows.append(
                {
                    "scenario": key,
                    "time": timestamp.isoformat(),
                    "target": target,
                    "observed": float(observations[target].values.ravel()[i]),
                    "initial": float(initial_output[target].values.ravel()[i]),
                    "optimized": float(optimized_output[target].values.ravel()[i]),
                }
            )
    return pd.DataFrame(rows)


def _parameter_frame(
    *,
    key: str,
    result: OptimizationResult,
    true_parameters: Mapping[str, float],
) -> pd.DataFrame:
    rows = []
    for name, initial in result.parameters_initial.items():
        rows.append(
            {
                "scenario": key,
                "parameter": name,
                "initial": float(initial),
                "optimized": float(result.parameters_optimized[name]),
                "true": float(true_parameters.get(name, np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _plot_single_target(
    case: Mapping[str, Any],
    *,
    target: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt, _ = _plotting_modules()
    frame = case["timeseries"].query("target == @target")
    fig, (ax, bar_ax) = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.2),
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )
    _plot_fit_lines(ax, frame, ylabel=ylabel)
    _plot_case_parameters(bar_ax, case["parameters"])
    fig.suptitle(case["title"], fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_svg(fig, output_path)
    plt.close(fig)


def _plot_energy_balance(case: Mapping[str, Any], output_path: Path) -> None:
    plt, _ = _plotting_modules()
    targets = ["F740", "Loutt", "LE"]
    fig, axes = plt.subplots(len(targets), 1, figsize=(9.0, 8.5), sharex=True)
    for ax, target in zip(axes, targets):
        frame = case["timeseries"].query("target == @target")
        _plot_fit_lines(ax, frame, ylabel=target)
    axes[0].set_title(case["title"], fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_svg(fig, output_path)
    plt.close(fig)


def _plot_parameter_summary(parameters: pd.DataFrame, output_path: Path) -> None:
    plt, _ = _plotting_modules()
    scenarios = list(parameters["scenario"].unique())
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(9.5, 8.5))
    for ax, scenario in zip(np.atleast_1d(axes), scenarios):
        subset = parameters[parameters["scenario"] == scenario].copy()
        labels = subset["parameter"].to_list()
        x = np.arange(len(labels))
        width = 0.25
        ax.bar(x - width, subset["initial"], width=width, label="Initial", color="#7a869a")
        ax.bar(x, subset["optimized"], width=width, label="Optimized", color="#00796b")
        ax.bar(x + width, subset["true"], width=width, label="Generating value", color="#f9ab00")
        ax.set_title(scenario.replace("_", " ").title())
        ax.set_xticks(x, labels)
        ax.grid(axis="y", color="#d9dee7", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    _save_svg(fig, output_path)
    plt.close(fig)


def _plot_fit_lines(ax: Any, frame: pd.DataFrame, *, ylabel: str) -> None:
    dates = pd.to_datetime(frame["time"])
    ax.plot(dates, frame["observed"], marker="o", label="Observed", color="#222222", linewidth=2.2)
    ax.plot(dates, frame["initial"], marker="s", label="Initial", color="#b44d12", linewidth=1.8)
    ax.plot(dates, frame["optimized"], marker="^", label="Optimized", color="#00796b", linewidth=1.8)
    ax.set_ylabel(ylabel)
    ax.grid(color="#d9dee7", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)


def _plot_case_parameters(ax: Any, parameters: pd.DataFrame) -> None:
    labels = parameters["parameter"].to_list()
    x = np.arange(len(labels))
    width = 0.25
    ax.bar(x - width, parameters["initial"], width=width, color="#7a869a", label="Initial")
    ax.bar(x, parameters["optimized"], width=width, color="#00796b", label="Optimized")
    ax.bar(x + width, parameters["true"], width=width, color="#f9ab00", label="Generating value")
    ax.set_xticks(x, labels)
    ax.set_title("Parameters")
    ax.grid(axis="y", color="#d9dee7", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def _save_svg(fig: Any, output_path: Path) -> None:
    """Save an SVG and normalize generated trailing whitespace."""
    fig.savefig(output_path, format="svg")
    text = output_path.read_text(encoding="utf-8")
    normalized = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    output_path.write_text(normalized, encoding="utf-8")


def _plotting_modules() -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt, matplotlib


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    files = generate_optimization_examples(args.output_dir)
    print("Wrote optimisation example artifacts:")
    for name, path in files.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
