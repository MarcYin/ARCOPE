"""Full ARC -> SCOPE experiment runner.

This module powers the repo's heavyweight, real-dependency example:

1. retrieve ARC biophysical parameters for a field and season
2. bridge them to SCOPE-ready inputs
3. fetch meteorological forcing
4. prepare SCOPE input datasets
5. run one or more SCOPE workflows from the shared retrieval
6. emit a docs-grade artifact bundle with figures and a markdown report

The default target uses the bundled Belgium test field for calendar year 2021
and documents the real ARC-to-SCOPE reflectance run. The module path keeps the
historical ``dual_workflow`` name for compatibility with earlier docs and
scripts.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from arc_scope.data import SHOWCASE_WEATHER_CSV, TEST_FIELD_GEOJSON
from arc_scope.pipeline.config import PipelineConfig
from arc_scope.pipeline.steps import (
    ArcResult,
    bridge_arc_to_scope,
    build_observation_dataset,
    fetch_weather,
    prepare_scope_dataset,
    retrieve_arc,
    run_scope_simulation,
)
from arc_scope.utils.io import load_geojson_bounds, load_geojson_centroid
from arc_scope.utils.types import PathLike

DEFAULT_START_DATE = "2021-05-15"
DEFAULT_END_DATE = "2021-10-01"
DEFAULT_WORKFLOWS = ("reflectance",)
DEFAULT_CROP_TYPE = "wheat"
DEFAULT_START_OF_SEASON = 170
DEFAULT_GROWTH_SEASON_LENGTH = 60
DEFAULT_NUM_SAMPLES = 100000
DEFAULT_LOCAL_VAR_MAP = {
    "sw_down_wm2": "Rin",
    "lw_down_wm2": "Rli",
    "air_temp_c": "Ta",
    "vapour_pressure_hpa": "ea",
    "pressure_hpa": "p",
    "wind_speed_ms": "u",
}
WORKFLOW_PRIORITY_PATTERNS = {
    "reflectance": (
        "rsot",
        "rso",
        "leaf_refl",
        "leaf_tran",
        "rdot",
        "rddt",
        "albedo",
        "reflect",
    ),
    "fluorescence": (
        "sif",
        "f685",
        "f740",
        "fluor",
        "fs",
        "qf",
        "apar",
    ),
    "thermal": (
        "lst",
        "ltt",
        "trad",
        "tcu",
        "tch",
        "tsu",
        "tsh",
        "rn",
        "h",
        "le",
        "thermal",
    ),
}


@dataclass(frozen=True)
class RuntimeCheck:
    """Simple runtime availability report for the full experiment."""

    package_versions: dict[str, str]
    requirements: dict[str, str]
    scope_root: str | None


@dataclass
class WorkflowRun:
    """Prepared input and output datasets for one SCOPE workflow."""

    scope_input_ds: xr.Dataset
    scope_output_ds: xr.Dataset
    selected_output_variables: list[str]


@dataclass
class DualWorkflowExperimentResult:
    """Shared and per-workflow products from the full experiment."""

    runtime: RuntimeCheck
    config: dict[str, Any]
    arc_result: ArcResult
    post_bio_da: xr.DataArray
    post_bio_scale_da: xr.DataArray
    weather_ds: xr.Dataset
    observation_ds: xr.Dataset
    acquisition_table: pd.DataFrame
    workflow_metrics: pd.DataFrame
    variable_inventory: pd.DataFrame
    workflow_runs: dict[str, WorkflowRun]


def run_dual_workflow_experiment(
    *,
    geojson_path: PathLike = TEST_FIELD_GEOJSON,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    crop_type: str = DEFAULT_CROP_TYPE,
    start_of_season: int = DEFAULT_START_OF_SEASON,
    year: int = 2021,
    workflows: Sequence[str] = DEFAULT_WORKFLOWS,
    output_dir: PathLike = Path("./full-run-output"),
    num_samples: int = DEFAULT_NUM_SAMPLES,
    growth_season_length: int = DEFAULT_GROWTH_SEASON_LENGTH,
    s2_data_folder: PathLike | None = None,
    data_source: str = "aws",
    weather_provider: str = "era5",
    weather_config: Mapping[str, Any] | None = None,
    scope_root_path: PathLike | None = None,
    device: str = "cpu",
    dtype: str = "float64",
) -> DualWorkflowExperimentResult:
    """Run the real ARC-SCOPE experiment from one shared retrieval."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    runtime = collect_runtime_check(
        weather_provider=weather_provider,
        scope_root_path=scope_root_path,
    )
    require_runtime_ready(runtime)

    weather_cfg = _resolve_weather_config(
        weather_provider=weather_provider,
        weather_config=weather_config,
    )
    workflows = tuple(dict.fromkeys(workflows))
    if not workflows:
        raise ValueError("Provide at least one SCOPE workflow.")

    base_config = PipelineConfig(
        geojson_path=geojson_path,
        start_date=start_date,
        end_date=end_date,
        crop_type=crop_type,
        start_of_season=start_of_season,
        year=year,
        num_samples=num_samples,
        growth_season_length=growth_season_length,
        s2_data_folder=s2_data_folder,
        data_source=data_source,
        weather_provider=weather_provider,
        weather_config=dict(weather_cfg),
        scope_workflow=workflows[0],
        scope_root_path=scope_root_path,
        output_dir=output_path,
        save_arc_npz=True,
        save_scope_netcdf=False,
        device=device,
        dtype=dtype,
    )

    config_summary = _build_run_config_summary(
        base_config=base_config,
        workflows=workflows,
        scope_root_path=runtime.scope_root,
    )

    arc_result = retrieve_arc(base_config)
    post_bio_da, post_bio_scale_da = bridge_arc_to_scope(arc_result, year=year)
    weather_ds = fetch_weather(base_config)
    observation_ds = build_observation_dataset(
        doys=arc_result.doys,
        year=year,
        geojson_path=geojson_path,
    )
    acquisition_table = _build_acquisition_table(arc_result, observation_ds)

    workflow_runs: dict[str, WorkflowRun] = {}
    variable_rows: list[dict[str, Any]] = []
    workflow_rows: list[dict[str, Any]] = []

    for workflow in workflows:
        workflow_config = _clone_pipeline_config(
            base_config,
            scope_workflow=workflow,
        )
        scope_input_ds = prepare_scope_dataset(
            post_bio_da=post_bio_da,
            post_bio_scale_da=post_bio_scale_da,
            weather_ds=weather_ds,
            observation_ds=observation_ds,
            config=workflow_config,
        )
        scope_output_ds = run_scope_simulation(scope_input_ds, workflow_config)
        selected_output_vars = select_workflow_variables(scope_output_ds, workflow)
        workflow_runs[workflow] = WorkflowRun(
            scope_input_ds=scope_input_ds,
            scope_output_ds=scope_output_ds,
            selected_output_variables=selected_output_vars,
        )

        variable_rows.extend(
            _summarize_dataset(scope_input_ds, dataset_name="scope_input", workflow=workflow)
        )
        variable_rows.extend(
            _summarize_dataset(scope_output_ds, dataset_name="scope_output", workflow=workflow)
        )
        workflow_rows.append(
            {
                "workflow": workflow,
                "n_input_vars": len(scope_input_ds.data_vars),
                "n_output_vars": len(scope_output_ds.data_vars),
                "n_time_steps": int(scope_output_ds.sizes.get("time", 0)),
                "selected_outputs": ";".join(selected_output_vars),
            }
        )

    variable_inventory = pd.DataFrame(variable_rows)
    workflow_metrics = pd.DataFrame(workflow_rows)
    return DualWorkflowExperimentResult(
        runtime=runtime,
        config=config_summary,
        arc_result=arc_result,
        post_bio_da=post_bio_da,
        post_bio_scale_da=post_bio_scale_da,
        weather_ds=weather_ds,
        observation_ds=observation_ds,
        acquisition_table=acquisition_table,
        workflow_metrics=workflow_metrics,
        variable_inventory=variable_inventory,
        workflow_runs=workflow_runs,
    )


def write_dual_workflow_artifacts(
    result: DualWorkflowExperimentResult,
    output_dir: PathLike,
) -> dict[str, Path]:
    """Write the full artifact bundle, figures, and markdown report."""
    output_path = Path(output_dir)
    figure_dir = output_path / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "run_config": output_path / "run_config.json",
        "environment": output_path / "environment.json",
        "post_bio": output_path / "post_bio.nc",
        "post_bio_scale": output_path / "post_bio_scale.nc",
        "weather": output_path / "weather.nc",
        "observation": output_path / "observation.nc",
        "acquisition_table": output_path / "acquisition_table.csv",
        "workflow_metrics": output_path / "workflow_metrics.csv",
        "variable_inventory": output_path / "variable_inventory.csv",
        "manifest": output_path / "artifact_manifest.json",
        "report": output_path / "index.md",
        "field_boundary": figure_dir / "field_boundary.png",
        "acquisition_timeline": figure_dir / "acquisition_timeline.svg",
        "weather_forcing": figure_dir / "weather_forcing.svg",
        "observation_geometry": figure_dir / "observation_geometry.svg",
        "arc_biophysics": figure_dir / "arc_biophysics.svg",
        "arc_peak_maps": figure_dir / "arc_peak_maps.png",
        "scope_input_overview": figure_dir / "scope_input_overview.svg",
    }
    arc_output_path = output_path / "arc_output.npz"
    if arc_output_path.exists():
        files["arc_output"] = arc_output_path

    files["run_config"].write_text(json.dumps(result.config, indent=2), encoding="utf-8")
    files["environment"].write_text(
        json.dumps(asdict(result.runtime), indent=2),
        encoding="utf-8",
    )
    _write_dataarray(result.post_bio_da, files["post_bio"], "post_bio")
    _write_dataarray(result.post_bio_scale_da, files["post_bio_scale"], "post_bio_scale")
    _write_dataset(result.weather_ds, files["weather"])
    _write_dataset(result.observation_ds, files["observation"])
    result.acquisition_table.to_csv(files["acquisition_table"], index=False)
    result.workflow_metrics.to_csv(files["workflow_metrics"], index=False)
    result.variable_inventory.to_csv(files["variable_inventory"], index=False)

    manifest: dict[str, str] = {}

    for workflow, workflow_run in result.workflow_runs.items():
        input_path = output_path / f"scope_input_{workflow}.nc"
        output_path_nc = output_path / f"scope_output_{workflow}.nc"
        ts_path = figure_dir / f"{workflow}_outputs.svg"
        map_path = figure_dir / f"{workflow}_snapshot_maps.png"
        files[f"scope_input_{workflow}"] = input_path
        files[f"scope_output_{workflow}"] = output_path_nc
        files[f"{workflow}_outputs"] = ts_path
        files[f"{workflow}_snapshot_maps"] = map_path

        _write_dataset(workflow_run.scope_input_ds, input_path)
        _write_dataset(workflow_run.scope_output_ds, output_path_nc)
        _plot_workflow_output_timeseries(
            workflow=workflow,
            workflow_run=workflow_run,
            path=ts_path,
        )
        _plot_workflow_snapshot_maps(
            workflow=workflow,
            workflow_run=workflow_run,
            peak_time=_peak_time_from_lai(result.post_bio_da),
            path=map_path,
        )

    _plot_field_boundary(
        geojson_path=Path(result.config["geojson_path"]),
        path=files["field_boundary"],
    )
    _plot_acquisition_timeline(
        acquisition_table=result.acquisition_table,
        path=files["acquisition_timeline"],
    )
    _plot_weather_forcing(result.weather_ds, files["weather_forcing"])
    _plot_observation_geometry(result.observation_ds, files["observation_geometry"])
    _plot_arc_biophysics(result.post_bio_da, files["arc_biophysics"])
    _plot_arc_peak_maps(result.post_bio_da, files["arc_peak_maps"])
    _plot_scope_input_overview(
        first_workflow=result.workflow_runs[next(iter(result.workflow_runs))].scope_input_ds,
        path=files["scope_input_overview"],
    )
    if len(result.workflow_runs) > 1:
        files["workflow_comparison"] = figure_dir / "workflow_comparison.svg"
        _plot_workflow_comparison(result.workflow_runs, files["workflow_comparison"])

    for key, path in files.items():
        manifest[key] = str(path.relative_to(Path(output_dir)))

    files["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files["report"].write_text(_render_report(result, manifest), encoding="utf-8")
    return files


def collect_runtime_check(
    *,
    weather_provider: str,
    scope_root_path: PathLike | None,
) -> RuntimeCheck:
    """Collect runtime availability and version details."""
    package_versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "arc_scope": _package_version("arc_scope"),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "xarray": _package_version("xarray"),
        "matplotlib": _package_version("matplotlib"),
    }
    requirements = {
        "arc": _module_status("arc"),
        "eof": _module_status("eof"),
        "osgeo": _module_status("osgeo"),
        "scope": _module_status("scope"),
        "torch": _module_status("torch"),
        "cdsapi": "not-required",
    }
    if weather_provider == "era5":
        requirements["cdsapi"] = _module_status("cdsapi")

    resolved_scope_root: str | None = None
    if requirements["scope"] == "available":
        try:
            from scope.spectral.loaders import scope_root

            resolved_scope_root = str(scope_root(scope_root_path))
        except Exception as exc:  # pragma: no cover - exercised by runtime only
            requirements["scope_root"] = f"missing:{exc}"
        else:
            requirements["scope_root"] = "available"
    else:
        requirements["scope_root"] = "blocked-by-scope"

    return RuntimeCheck(
        package_versions=package_versions,
        requirements=requirements,
        scope_root=resolved_scope_root,
    )


def require_runtime_ready(runtime: RuntimeCheck) -> None:
    """Raise a clear runtime error when full dependencies are unavailable."""
    blockers = [
        f"{name}: {status}"
        for name, status in runtime.requirements.items()
        if status not in {"available", "not-required"}
    ]
    if blockers:
        raise RuntimeError(
            "The full ARC-to-SCOPE experiment requires a live ARC + SCOPE runtime. "
            "Fix the following prerequisites first: " + "; ".join(blockers)
        )


def select_workflow_variables(ds: xr.Dataset, workflow: str, limit: int = 4) -> list[str]:
    """Choose a small set of high-signal output variables for plotting."""
    candidates = [
        name for name in ds.data_vars
        if _is_numeric(ds[name]) and "time" in ds[name].dims
    ]
    selected: list[str] = []
    for pattern in WORKFLOW_PRIORITY_PATTERNS.get(workflow, ()):
        for name in candidates:
            if name in selected:
                continue
            if pattern in name.lower():
                selected.append(name)
            if len(selected) >= limit:
                return selected

    for name in sorted(candidates):
        if name not in selected:
            selected.append(name)
        if len(selected) >= limit:
            break
    return selected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the full example runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geojson-path", type=Path, default=TEST_FIELD_GEOJSON)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--crop-type", default=DEFAULT_CROP_TYPE)
    parser.add_argument("--start-of-season", type=int, default=DEFAULT_START_OF_SEASON)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument(
        "--workflow",
        dest="workflows",
        action="append",
        default=[],
        help="Workflow to run. Repeat for multiple values. Default: reflectance.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./full-run-output"))
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument(
        "--growth-season-length",
        type=int,
        default=DEFAULT_GROWTH_SEASON_LENGTH,
    )
    parser.add_argument("--s2-data-folder", type=Path, default=None)
    parser.add_argument("--data-source", default="aws")
    parser.add_argument(
        "--weather-provider",
        choices=("era5", "local"),
        default="era5",
    )
    parser.add_argument("--weather-file", type=Path, default=None)
    parser.add_argument("--scope-root-path", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument(
        "--check-runtime",
        action="store_true",
        help="Print the runtime availability report and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the full ARC-SCOPE example from the command line."""
    args = parse_args(argv)
    workflows = tuple(args.workflows or DEFAULT_WORKFLOWS)
    weather_config = None
    if args.weather_provider == "local":
        local_weather_file = args.weather_file or SHOWCASE_WEATHER_CSV
        weather_config = _resolve_weather_config(
            weather_provider="local",
            weather_config={"file_path": local_weather_file},
        )

    runtime = collect_runtime_check(
        weather_provider=args.weather_provider,
        scope_root_path=args.scope_root_path,
    )
    if args.check_runtime:
        print(json.dumps(asdict(runtime), indent=2))
        return

    result = run_dual_workflow_experiment(
        geojson_path=args.geojson_path,
        start_date=args.start_date,
        end_date=args.end_date,
        crop_type=args.crop_type,
        start_of_season=args.start_of_season,
        year=args.year,
        workflows=workflows,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        growth_season_length=args.growth_season_length,
        s2_data_folder=args.s2_data_folder,
        data_source=args.data_source,
        weather_provider=args.weather_provider,
        weather_config=weather_config,
        scope_root_path=args.scope_root_path,
        device=args.device,
        dtype=args.dtype,
    )
    files = write_dual_workflow_artifacts(result, args.output_dir)

    print("=" * 72)
    print("ARC-SCOPE End-to-End Experiment")
    print("=" * 72)
    print(f"Field:        {args.geojson_path}")
    print(f"Year:         {args.year}")
    print(f"Date window:  {args.start_date} -> {args.end_date}")
    print(f"Workflows:    {', '.join(workflows)}")
    print(f"Scope root:   {result.runtime.scope_root}")
    print()
    print("Artifacts written:")
    for name in sorted(files):
        print(f"  {name:>20s}: {files[name]}")


def _resolve_weather_config(
    *,
    weather_provider: str,
    weather_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Normalise weather configuration with sensible example defaults."""
    cfg = dict(weather_config or {})
    if weather_provider == "local":
        cfg.setdefault("file_path", SHOWCASE_WEATHER_CSV)
        cfg.setdefault("var_map", DEFAULT_LOCAL_VAR_MAP)
        cfg.setdefault("time_column", "time")
    return cfg


def _clone_pipeline_config(config: PipelineConfig, *, scope_workflow: str) -> PipelineConfig:
    """Copy a pipeline config while changing the SCOPE workflow."""
    return PipelineConfig(
        geojson_path=config.geojson_path,
        start_date=config.start_date,
        end_date=config.end_date,
        crop_type=config.crop_type,
        start_of_season=config.start_of_season,
        year=config.year,
        num_samples=config.num_samples,
        growth_season_length=config.growth_season_length,
        s2_data_folder=config.s2_data_folder,
        data_source=config.data_source,
        weather_provider=config.weather_provider,
        weather_config=dict(config.weather_config),
        scope_workflow=scope_workflow,
        scope_root_path=config.scope_root_path,
        scope_options=dict(config.scope_options),
        device=config.device,
        dtype=config.dtype,
        output_dir=config.output_dir,
        save_arc_npz=config.save_arc_npz,
        save_scope_netcdf=config.save_scope_netcdf,
        optimize=config.optimize,
        optim_config=config.optim_config,
    )


def _build_run_config_summary(
    *,
    base_config: PipelineConfig,
    workflows: Sequence[str],
    scope_root_path: str | None,
) -> dict[str, Any]:
    """Create a JSON-serialisable configuration summary."""
    lon, lat = load_geojson_centroid(base_config.geojson_path)
    bounds = load_geojson_bounds(base_config.geojson_path)
    return {
        "geojson_path": str(base_config.geojson_path),
        "location": {
            "longitude": lon,
            "latitude": lat,
            "bounds_wgs84": list(bounds),
        },
        "start_date": base_config.start_date,
        "end_date": base_config.end_date,
        "crop_type": base_config.crop_type,
        "start_of_season": base_config.start_of_season,
        "year": base_config.year,
        "num_samples": base_config.num_samples,
        "growth_season_length": base_config.growth_season_length,
        "data_source": base_config.data_source,
        "weather_provider": base_config.weather_provider,
        "weather_config": _stringify_paths(base_config.weather_config),
        "workflows": list(workflows),
        "scope_root_path": scope_root_path,
        "device": base_config.device,
        "dtype": base_config.dtype,
        "python": sys.version.split()[0],
    }


def _build_acquisition_table(
    arc_result: ArcResult,
    observation_ds: xr.Dataset,
) -> pd.DataFrame:
    """Create a flat acquisition table for dates and geometry."""
    times = pd.to_datetime(observation_ds.coords["time"].values)
    return pd.DataFrame(
        {
            "doy": arc_result.doys.astype(int),
            "date": times.strftime("%Y-%m-%d"),
            "solar_zenith_angle": observation_ds["solar_zenith_angle"].values,
            "solar_azimuth_angle": observation_ds["solar_azimuth_angle"].values,
            "viewing_zenith_angle": observation_ds["viewing_zenith_angle"].values,
            "viewing_azimuth_angle": observation_ds["viewing_azimuth_angle"].values,
        }
    )


def _summarize_dataset(
    ds: xr.Dataset,
    *,
    dataset_name: str,
    workflow: str,
) -> list[dict[str, Any]]:
    """Summarise dataset variables for audit-friendly CSV output."""
    rows: list[dict[str, Any]] = []
    for name, da in ds.data_vars.items():
        if not _is_numeric(da):
            continue
        values = np.asarray(da.values, dtype=np.float64)
        finite_mask = np.isfinite(values)
        if finite_mask.any():
            finite = values[finite_mask]
            min_value = float(finite.min())
            max_value = float(finite.max())
            mean_value = float(finite.mean())
        else:
            min_value = np.nan
            max_value = np.nan
            mean_value = np.nan
        rows.append(
            {
                "workflow": workflow,
                "dataset": dataset_name,
                "variable": name,
                "dims": "|".join(da.dims),
                "nan_fraction": float(1.0 - finite_mask.mean()),
                "min": min_value,
                "max": max_value,
                "mean": mean_value,
            }
        )
    return rows


def _plot_field_boundary(geojson_path: Path, path: Path) -> None:
    """Render the field polygon(s) in geographic coordinates."""
    with geojson_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    fig, ax = plt.subplots(figsize=(6, 6))
    for ring in _iter_polygon_rings(payload):
        xs = [point[0] for point in ring]
        ys = [point[1] for point in ring]
        ax.fill(xs, ys, alpha=0.18, color="#0f766e")
        ax.plot(xs, ys, color="#0f766e", linewidth=2)
    ax.set_title("Field boundary used for ARC retrieval")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_acquisition_timeline(acquisition_table: pd.DataFrame, path: Path) -> None:
    """Plot the Sentinel-2 acquisition timeline used for ARC assimilation."""
    times = pd.to_datetime(acquisition_table["date"])
    fig, ax = plt.subplots(figsize=(9, 3.5))
    markerline, stemlines, baseline = ax.stem(times, acquisition_table["doy"])
    plt.setp(markerline, color="#1d4ed8", markersize=6)
    plt.setp(stemlines, color="#1d4ed8", linewidth=1.5)
    plt.setp(baseline, color="#94a3b8", linewidth=1.0)
    ax.set_title("Acquisition timeline for the ARC retrieval")
    ax.set_ylabel("Day of year")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_weather_forcing(weather_ds: xr.Dataset, path: Path) -> None:
    """Plot the key forcing variables used by SCOPE."""
    variables = ("Rin", "Rli", "Ta", "u")
    colors = {
        "Rin": "#ea580c",
        "Rli": "#7c3aed",
        "Ta": "#dc2626",
        "u": "#0369a1",
    }
    titles = {
        "Rin": "Incoming shortwave",
        "Rli": "Incoming longwave",
        "Ta": "Air temperature",
        "u": "Wind speed",
    }
    units = {
        "Rin": "W m-2",
        "Rli": "W m-2",
        "Ta": "degC",
        "u": "m s-1",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    axes = axes.ravel()
    for ax, name in zip(axes, variables):
        if name not in weather_ds:
            ax.set_visible(False)
            continue
        ax.plot(weather_ds["time"].values, weather_ds[name].values, color=colors[name], linewidth=2)
        ax.set_title(titles[name])
        ax.set_ylabel(units[name])
        ax.grid(alpha=0.25)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Meteorological forcing passed into SCOPE", y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_observation_geometry(observation_ds: xr.Dataset, path: Path) -> None:
    """Plot solar and viewing geometry through the season."""
    time = pd.to_datetime(observation_ds.coords["time"].values)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(time, observation_ds["solar_zenith_angle"], label="Solar zenith", color="#f59e0b", linewidth=2)
    axes[0].plot(time, observation_ds["viewing_zenith_angle"], label="Viewing zenith", color="#1d4ed8", linewidth=2)
    axes[0].set_ylabel("Degrees")
    axes[0].set_title("Zenith geometry")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(time, observation_ds["solar_azimuth_angle"], label="Solar azimuth", color="#b45309", linewidth=2)
    axes[1].plot(time, observation_ds["viewing_azimuth_angle"], label="Viewing azimuth", color="#0369a1", linewidth=2)
    axes[1].set_ylabel("Degrees")
    axes[1].set_title("Azimuth geometry")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[1].tick_params(axis="x", rotation=30)

    fig.suptitle("Observation geometry derived from the field centroid", y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_arc_biophysics(post_bio_da: xr.DataArray, path: Path) -> None:
    """Plot spatially averaged ARC biophysical trajectories."""
    bands = ("lai", "cab", "cw")
    colors = {"lai": "#15803d", "cab": "#16a34a", "cw": "#0f766e"}
    units = {"lai": "m2 m-2", "cab": "ug cm-2", "cw": "g cm-2"}
    time = pd.to_datetime(post_bio_da.coords["time"].values)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, band in zip(axes, bands):
        series = post_bio_da.sel(band=band).mean(dim=("y", "x"), skipna=True)
        ax.plot(time, series.values, color=colors[band], linewidth=2)
        ax.set_title(f"ARC posterior mean: {band}")
        ax.set_ylabel(units[band])
        ax.grid(alpha=0.25)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[-1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_arc_peak_maps(post_bio_da: xr.DataArray, path: Path) -> None:
    """Plot maps for key ARC variables at peak mean LAI."""
    peak_time = _peak_time_from_lai(post_bio_da)
    bands = ("lai", "cab", "cw")
    titles = {
        "lai": "LAI at peak canopy",
        "cab": "Cab at peak canopy",
        "cw": "Cw at peak canopy",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, band in zip(axes, bands):
        image = post_bio_da.sel(time=peak_time, band=band).transpose("y", "x")
        mappable = ax.imshow(image.values, cmap="viridis")
        ax.set_title(titles[band])
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(mappable, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_scope_input_overview(first_workflow: xr.Dataset, path: Path) -> None:
    """Plot a representative subset of the prepared SCOPE inputs."""
    variables = ("LAI", "Cab", "Cw", "Ta", "Rin", "tts")
    titles = {
        "LAI": "Leaf area index",
        "Cab": "Leaf chlorophyll",
        "Cw": "Leaf water",
        "Ta": "Air temperature",
        "Rin": "Incoming shortwave",
        "tts": "Solar zenith angle",
    }
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    axes = axes.ravel()
    for ax, name in zip(axes, variables):
        series = _reduce_to_time_series(first_workflow[name])
        ax.plot(first_workflow["time"].values, series.values, linewidth=2, color="#334155")
        ax.set_title(titles[name])
        ax.grid(alpha=0.25)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Representative SCOPE inputs used in the real run", y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_workflow_output_timeseries(
    *,
    workflow: str,
    workflow_run: WorkflowRun,
    path: Path,
) -> None:
    """Plot high-signal time-series outputs for one workflow."""
    variables = workflow_run.selected_output_variables
    ncols = 2
    nrows = int(np.ceil(len(variables) / ncols)) or 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.8 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    time = pd.to_datetime(workflow_run.scope_output_ds["time"].values)
    for ax, name in zip(axes, variables):
        series = _reduce_to_time_series(workflow_run.scope_output_ds[name])
        ax.plot(time, series.values, linewidth=2, color="#0f766e")
        ax.set_title(name)
        ax.grid(alpha=0.25)
    for ax in axes[len(variables):]:
        ax.set_visible(False)
    if len(axes):
        axes[min(len(variables), len(axes)) - 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"{_workflow_label(workflow)} output time series", y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_workflow_snapshot_maps(
    *,
    workflow: str,
    workflow_run: WorkflowRun,
    peak_time: pd.Timestamp,
    path: Path,
) -> None:
    """Plot spatial maps for the selected workflow outputs."""
    map_variables = [
        name for name in workflow_run.selected_output_variables
        if _can_reduce_to_map(workflow_run.scope_output_ds[name])
    ][:3]
    if not map_variables:
        fig, ax = plt.subplots(figsize=(7, 2.4))
        ax.text(
            0.5,
            0.5,
            f"No y/x output maps available for {workflow}.",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, len(map_variables), figsize=(4 * len(map_variables), 4.2))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, map_variables):
        image = _reduce_to_map(workflow_run.scope_output_ds[name], peak_time)
        mappable = ax.imshow(image.values, cmap="magma")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(mappable, ax=ax, shrink=0.8)
    fig.suptitle(f"{_workflow_label(workflow)} output maps at peak LAI", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_workflow_comparison(workflow_runs: Mapping[str, WorkflowRun], path: Path) -> None:
    """Compare common time-series outputs across workflows."""
    workflows = list(workflow_runs)
    shared = set(workflow_runs[workflows[0]].scope_output_ds.data_vars)
    for workflow in workflows[1:]:
        shared &= set(workflow_runs[workflow].scope_output_ds.data_vars)

    comparable = [
        name for name in sorted(shared)
        if all(_is_numeric(workflow_runs[workflow].scope_output_ds[name]) for workflow in workflows)
        and all("time" in workflow_runs[workflow].scope_output_ds[name].dims for workflow in workflows)
    ][:4]
    if not comparable:
        fig, ax = plt.subplots(figsize=(7, 2.6))
        ax.text(
            0.5,
            0.5,
            "The workflows do not expose a shared time-series output set to compare directly.",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    fig, axes = plt.subplots(len(comparable), 1, figsize=(10, 3.2 * len(comparable)), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    colors = ["#1d4ed8", "#dc2626", "#0f766e", "#7c3aed"]
    for ax, name in zip(axes, comparable):
        for idx, workflow in enumerate(workflows):
            series = _reduce_to_time_series(workflow_runs[workflow].scope_output_ds[name])
            ax.plot(
                workflow_runs[workflow].scope_output_ds["time"].values,
                series.values,
                linewidth=2,
                color=colors[idx % len(colors)],
                label=workflow,
            )
        ax.set_title(name)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Shared outputs compared across workflows", y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_report(result: DualWorkflowExperimentResult, manifest: Mapping[str, str]) -> str:
    """Render a markdown report beside the generated artifact bundle."""
    workflow_names = list(result.workflow_runs)
    workflow_summary = ", ".join(_workflow_label(name) for name in workflow_names)
    has_comparison = len(workflow_names) > 1
    lines = [
        "# ARC-SCOPE End-to-End Experiment",
        "",
        "## Scenario",
        "",
        f"- Field: `{result.config['geojson_path']}`",
        f"- Location: `{result.config['location']['latitude']:.5f}N, {result.config['location']['longitude']:.5f}E`",
        f"- Date window: `{result.config['start_date']}` to `{result.config['end_date']}`",
        f"- Crop type: `{result.config['crop_type']}`",
        f"- SCOPE workflow{'s' if len(workflow_names) > 1 else ''}: `{', '.join(workflow_names)}`",
        "",
        "This run couples two model stages:",
        "",
        "1. ARC retrieves crop biophysical state from Sentinel-2 acquisitions over the field and season.",
        f"2. SCOPE simulates {_pluralise('output', len(workflow_names))} for the {workflow_summary.lower()} workflow{'s' if len(workflow_names) > 1 else ''} using the ARC state, meteorological forcing, and observation geometry.",
        "",
        "The heavyweight example currently documents the prepared, runnable reflectance-backed experiment path. Other SCOPE workflows may exist elsewhere in the package surface, but they are not the primary documented contract of this report.",
        "",
        "## What This Run Produces",
        "",
        "- `arc_output.npz` from ARC retrieval plus bridged `post_bio.nc` and `post_bio_scale.nc`.",
        "- Shared forcing datasets: `weather.nc`, `observation.nc`, and `acquisition_table.csv`.",
        "- SCOPE-ready experiment inputs plus one NetCDF output dataset per workflow.",
        "- A figure suite covering field geometry, acquisitions, forcing, retrieved crop state, SCOPE inputs, and simulated outputs.",
        "- `workflow_metrics.csv` and `variable_inventory.csv` to audit the variable coverage and data health.",
        "",
        "## Figures",
        "",
        "### Field and Observation Context",
        "",
        "![Field boundary](figures/field_boundary.png)",
        "",
        "The field boundary plot confirms the geographic target passed into ARC and the spatial footprint used for later map products.",
        "",
        "![Acquisition timeline](figures/acquisition_timeline.svg)",
        "",
        "The acquisition timeline shows the Sentinel-2 observations assimilated by ARC and the dates used to build the observation geometry.",
        "",
        "![Weather forcing](figures/weather_forcing.svg)",
        "",
        "The weather forcing panels show the atmospheric drivers that feed SCOPE. `Rin` and `Rli` drive radiative forcing, while `Ta` and `u` capture the atmospheric state used during simulation.",
        "",
        "![Observation geometry](figures/observation_geometry.svg)",
        "",
        "The observation geometry chart shows the solar and viewing angles derived from the field centroid for each acquisition date. These angles control the sun-sensor geometry seen by SCOPE.",
        "",
        "### ARC Retrieval And SCOPE Inputs",
        "",
        "![ARC biophysics](figures/arc_biophysics.svg)",
        "",
        "This panel summarises the seasonal trajectories of the key ARC posterior variables before they are converted to SCOPE inputs. `LAI` controls canopy density, `Cab` controls chlorophyll absorption, and `Cw` controls leaf water content.",
        "",
        "![ARC peak maps](figures/arc_peak_maps.png)",
        "",
        "The peak-date maps show how ARC's spatial heterogeneity propagates into the downstream SCOPE simulation. Spatial structure in these maps is the starting point for the simulated reflectance outputs.",
        "",
        "![SCOPE input overview](figures/scope_input_overview.svg)",
        "",
        "This figure combines the most important SCOPE inputs. `LAI`, `Cab`, and `Cw` come from ARC retrieval, `Ta` and `Rin` come from weather forcing, and `tts` shows the solar zenith angle used during the run.",
        "",
        "### Simulated Outputs",
        "",
    ]
    for workflow in workflow_names:
        selected = result.workflow_runs[workflow].selected_output_variables
        lines.extend(
            [
                f"#### {_workflow_label(workflow)}",
                "",
                f"![{_workflow_label(workflow)} outputs](figures/{workflow}_outputs.svg)",
                "",
                f"Time-series summaries for the `{workflow}` workflow outputs. This figure highlights the actual variables selected from the saved dataset: `{', '.join(selected)}`.",
                "",
                f"![{_workflow_label(workflow)} maps](figures/{workflow}_snapshot_maps.png)",
                "",
                f"Spatial snapshots for the `{workflow}` workflow at the peak-LAI date, reduced to y/x maps when the variables support it. This shows where the simulated canopy response is strongest across the field.",
                "",
            ]
        )
    if has_comparison:
        lines.extend(
            [
                "### Workflow Comparison",
                "",
                "![Workflow comparison](figures/workflow_comparison.svg)",
                "",
                "This comparison figure overlays the outputs that exist in each workflow result dataset so you can see where the model branches diverge or agree.",
                "",
            ]
        )
    lines.extend(
        [
            "## Key Files",
            "",
        ]
    )
    for key in (
        "run_config",
        "environment",
        "arc_output",
        "post_bio",
        "post_bio_scale",
        "weather",
        "observation",
        "acquisition_table",
        "workflow_metrics",
        "variable_inventory",
        "manifest",
    ):
        if key in manifest:
            lines.append(f"- [{key}]({manifest[key]})")
    for workflow in workflow_names:
        lines.append(f"- [scope_input_{workflow}]({manifest[f'scope_input_{workflow}']})")
        lines.append(f"- [scope_output_{workflow}]({manifest[f'scope_output_{workflow}']})")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The figure selections are data-driven. If a workflow exposes different variables, the report adapts rather than assuming a fixed output schema.",
            "- The report is generated from the saved experiment outputs so it stays aligned with the actual run and artifact names.",
            "- In the default documented path, the two models are ARC retrieval first and SCOPE reflectance simulation second.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_dataset(ds: xr.Dataset, path: Path) -> None:
    """Write a dataset to NetCDF with a practical, low-overhead fallback chain."""
    try:
        ds.to_netcdf(path, engine="netcdf4")
    except Exception:
        try:
            ds.to_netcdf(path, engine="h5netcdf")
        except Exception:
            ds.to_netcdf(path, engine="scipy")


def _write_dataarray(da: xr.DataArray, path: Path, name: str) -> None:
    """Write a named data array as a NetCDF dataset."""
    _write_dataset(da.to_dataset(name=name), path)


def _peak_time_from_lai(post_bio_da: xr.DataArray) -> pd.Timestamp:
    """Find the timestamp with the highest mean LAI."""
    lai_series = post_bio_da.sel(band="lai").mean(dim=("y", "x"), skipna=True)
    peak_index = int(lai_series.argmax(dim="time").item())
    return pd.Timestamp(post_bio_da["time"].values[peak_index])


def _reduce_to_time_series(da: xr.DataArray) -> xr.DataArray:
    """Collapse a variable to a time series by averaging extra dimensions."""
    reduced = da
    for dim in list(reduced.dims):
        if dim != "time":
            reduced = reduced.mean(dim=dim, skipna=True)
    return reduced


def _can_reduce_to_map(da: xr.DataArray) -> bool:
    """Whether a variable can be reduced to a y/x map."""
    reduced = da
    for dim in da.dims:
        if dim in {"time", "y", "x"}:
            continue
        reduced = reduced.mean(dim=dim, skipna=True)
    return set(reduced.dims) >= {"y", "x"}


def _reduce_to_map(da: xr.DataArray, time_value: pd.Timestamp) -> xr.DataArray:
    """Reduce a variable to a y/x map at one time."""
    reduced = da
    if "time" in reduced.dims:
        reduced = reduced.sel(time=np.datetime64(time_value), method="nearest")
    for dim in list(reduced.dims):
        if dim not in {"y", "x"}:
            reduced = reduced.mean(dim=dim, skipna=True)
    return reduced.transpose("y", "x")


def _iter_polygon_rings(payload: Mapping[str, Any]) -> list[list[list[float]]]:
    """Extract polygon rings from a GeoJSON-like structure."""
    geometry_type = payload.get("type")
    if geometry_type == "FeatureCollection":
        rings: list[list[list[float]]] = []
        for feature in payload.get("features", []):
            rings.extend(_iter_polygon_rings(feature.get("geometry", {})))
        return rings
    if geometry_type == "Feature":
        return _iter_polygon_rings(payload.get("geometry", {}))
    if geometry_type == "Polygon":
        return [ring for ring in payload.get("coordinates", [])]
    if geometry_type == "MultiPolygon":
        rings = []
        for polygon in payload.get("coordinates", []):
            rings.extend(polygon)
        return rings
    return []


def _is_numeric(da: xr.DataArray) -> bool:
    """Whether a data array holds numeric values."""
    return np.issubdtype(da.dtype, np.number)


def _module_status(module_name: str) -> str:
    """Return a compact import status string."""
    try:
        __import__(module_name)
    except Exception as exc:
        return f"missing:{exc}"
    return "available"


def _package_version(package_name: str) -> str:
    """Return a package version or ``unknown`` if unavailable."""
    try:
        try:
            from importlib.metadata import version
        except ImportError:  # pragma: no cover
            from importlib_metadata import version  # type: ignore
        return version(package_name)
    except Exception:
        return "unknown"


def _stringify_paths(value: Any) -> Any:
    """Convert paths inside nested structures to strings."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _stringify_paths(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify_paths(v) for v in value]
    return value


def _workflow_label(workflow: str) -> str:
    """Render a user-facing workflow label."""
    return workflow.replace("-", " ").title()


def _pluralise(word: str, count: int) -> str:
    """Return a simple singular/plural choice."""
    if count == 1:
        return word
    return f"{word}s"


WorkflowExperimentResult = DualWorkflowExperimentResult
run_full_experiment = run_dual_workflow_experiment
write_full_run_artifacts = write_dual_workflow_artifacts


if __name__ == "__main__":  # pragma: no cover
    main()
