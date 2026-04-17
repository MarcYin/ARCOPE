"""Reproducible showcase experiment built on ARC-SCOPE core dependencies.

This module intentionally stays within the repo's fully tested surfaces:

- synthetic ARC-like posterior arrays
- bridge conversion to SCOPE-style inputs
- bundled field geometry
- local weather ingestion
- radiation partitioning
- a clearly labeled proxy calibration step

The proxy calibration is not a substitute for a full ``scope-rtm`` run. It is
used here to provide a deterministic, end-to-end optimisation example that can
run on the core dependency set.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from arc_scope.bridge.convert import arc_arrays_to_scope_inputs
from arc_scope.bridge.parameter_map import BIO_BANDS, BIO_SCALES
from arc_scope.data import SHOWCASE_WEATHER_CSV, TEST_FIELD_GEOJSON
from arc_scope.optim.objective import ScopeObjective
from arc_scope.optim.parameters import ParameterSet, ParameterSpec
from arc_scope.optim.protocols import ScipyOptimizer
from arc_scope.pipeline.steps import build_observation_dataset
from arc_scope.utils.io import load_geojson_bounds
from arc_scope.utils.types import PathLike
from arc_scope.weather.local import LocalProvider
from arc_scope.weather.radiation import partition_shortwave

SHOWCASE_DOYS = np.array([145, 153, 161, 169, 177, 185, 193, 201, 209, 217], dtype=int)
SHOWCASE_YEAR = 2021
SHOWCASE_INITIAL_FQE = 0.008
SHOWCASE_TRUE_FQE = 0.018


@dataclass(frozen=True)
class ShowcaseSummary:
    """Compact metrics for the showcase experiment."""

    n_time_steps: int
    peak_lai: float
    mean_direct_fraction: float
    mean_diffuse_fraction: float
    true_fqe: float
    initial_fqe: float
    optimized_fqe: float
    rmse_initial: float
    rmse_optimized: float
    relative_fqe_error_pct: float


@dataclass
class ShowcaseExperimentResult:
    """Full result bundle returned by :func:`run_showcase_experiment`."""

    post_bio_da: xr.DataArray
    post_bio_scale_da: xr.DataArray
    weather_ds: xr.Dataset
    observation_ds: xr.Dataset
    experiment_ds: xr.Dataset
    observations_ds: xr.Dataset
    initial_output_ds: xr.Dataset
    fitted_output_ds: xr.Dataset
    timeseries: pd.DataFrame
    summary: ShowcaseSummary


def run_showcase_experiment(
    *,
    seed: int = 7,
    year: int = SHOWCASE_YEAR,
    initial_fqe: float = SHOWCASE_INITIAL_FQE,
    true_fqe: float = SHOWCASE_TRUE_FQE,
) -> ShowcaseExperimentResult:
    """Run the bundled showcase experiment.

    Parameters
    ----------
    seed:
        Random seed for deterministic observation noise.
    year:
        Calendar year combined with ``SHOWCASE_DOYS``.
    initial_fqe:
        Starting proxy fluorescence yield used by the optimiser.
    true_fqe:
        Hidden value used to generate synthetic observations.
    """
    doys = SHOWCASE_DOYS.copy()
    mask = _build_showcase_mask()
    post_bio_tensor, scale_data, geotransform = _build_showcase_arc_inputs(mask, doys)
    post_bio_da, post_bio_scale_da = arc_arrays_to_scope_inputs(
        post_bio_tensor=post_bio_tensor,
        scale_data=scale_data,
        mask=mask,
        doys=doys,
        geotransform=geotransform,
        crs="EPSG:4326",
        year=year,
    )

    observation_ds = build_observation_dataset(
        doys=doys,
        year=year,
        geojson_path=TEST_FIELD_GEOJSON,
    )
    weather_ds = _load_showcase_weather(observation_ds.coords["time"].values)
    experiment_ds = _build_proxy_experiment_dataset(
        post_bio_da=post_bio_da,
        weather_ds=weather_ds,
        observation_ds=observation_ds,
    )
    observations_ds = _build_proxy_observations(
        experiment_ds=experiment_ds,
        true_fqe=true_fqe,
        seed=seed,
    )

    objective = ScopeObjective(
        base_dataset=experiment_ds,
        observations=observations_ds,
        target_variables=["proxy_sif"],
        scope_runner=_run_proxy_fluorescence_model,
    )
    param_set = ParameterSet(
        [
            ParameterSpec(
                "fqe",
                initial=initial_fqe,
                lower=0.004,
                upper=0.05,
                transform="log",
            )
        ]
    )
    optimizer = ScipyOptimizer(method="L-BFGS-B", max_iter=60, tol=1e-12)

    initial_output_ds = _run_proxy_fluorescence_model(
        param_set.inject_into_dataset(experiment_ds, {"fqe": initial_fqe})
    )
    rmse_initial = float(
        np.sqrt(
            objective.evaluate({"fqe": initial_fqe})
        )
    )

    optimizer.step(objective, param_set)
    optimized_fqe = param_set.specs[0].initial
    fitted_output_ds = _run_proxy_fluorescence_model(
        param_set.inject_into_dataset(experiment_ds, {"fqe": optimized_fqe})
    )
    rmse_optimized = float(np.sqrt(objective.evaluate({"fqe": optimized_fqe})))

    timeseries = _build_timeseries_frame(
        experiment_ds=experiment_ds,
        observations_ds=observations_ds,
        initial_output_ds=initial_output_ds,
        fitted_output_ds=fitted_output_ds,
    )
    summary = ShowcaseSummary(
        n_time_steps=int(timeseries.shape[0]),
        peak_lai=float(timeseries["lai"].max()),
        mean_direct_fraction=float(timeseries["direct_fraction"].mean()),
        mean_diffuse_fraction=float(timeseries["diffuse_fraction"].mean()),
        true_fqe=float(true_fqe),
        initial_fqe=float(initial_fqe),
        optimized_fqe=float(optimized_fqe),
        rmse_initial=rmse_initial,
        rmse_optimized=rmse_optimized,
        relative_fqe_error_pct=float(abs(optimized_fqe - true_fqe) / true_fqe * 100.0),
    )

    return ShowcaseExperimentResult(
        post_bio_da=post_bio_da,
        post_bio_scale_da=post_bio_scale_da,
        weather_ds=weather_ds,
        observation_ds=observation_ds,
        experiment_ds=experiment_ds,
        observations_ds=observations_ds,
        initial_output_ds=initial_output_ds,
        fitted_output_ds=fitted_output_ds,
        timeseries=timeseries,
        summary=summary,
    )


def write_showcase_artifacts(
    result: ShowcaseExperimentResult,
    output_dir: PathLike,
) -> dict[str, Path]:
    """Write docs-ready artifacts for the showcase experiment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = {
        "summary": output_path / "summary.json",
        "timeseries": output_path / "timeseries.csv",
        "radiation_chart": output_path / "radiation_partition.svg",
        "proxy_sif_chart": output_path / "proxy_sif_fit.svg",
    }

    files["summary"].write_text(
        json.dumps(asdict(result.summary), indent=2),
        encoding="utf-8",
    )
    result.timeseries.to_csv(files["timeseries"], index=False)

    _write_line_chart_svg(
        path=files["radiation_chart"],
        x_values=result.timeseries["date"].tolist(),
        series={
            "Direct shortwave": result.timeseries["Rin_direct"].tolist(),
            "Diffuse shortwave": result.timeseries["Rin_diffuse"].tolist(),
        },
        colors={
            "Direct shortwave": "#c2410c",
            "Diffuse shortwave": "#2563eb",
        },
        title="Direct and diffuse shortwave forcing",
        y_label="Radiation (W m-2)",
    )
    _write_line_chart_svg(
        path=files["proxy_sif_chart"],
        x_values=result.timeseries["date"].tolist(),
        series={
            "Observed proxy SIF": result.timeseries["proxy_sif_observed"].tolist(),
            "Initial fit": result.timeseries["proxy_sif_initial"].tolist(),
            "Optimized fit": result.timeseries["proxy_sif_fitted"].tolist(),
        },
        colors={
            "Observed proxy SIF": "#111827",
            "Initial fit": "#dc2626",
            "Optimized fit": "#16a34a",
        },
        title="Proxy fluorescence calibration",
        y_label="Proxy SIF (a.u.)",
    )

    # Interactive HTML dashboard
    dashboard_path = output_path / "dashboard.html"
    _write_showcase_dashboard_html(result, dashboard_path)
    files["dashboard"] = dashboard_path

    return files


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the packaged showcase command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./showcase-output"),
        help="Directory where summary files and SVG charts will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Deterministic seed for the synthetic observation noise.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the showcase experiment as a packaged CLI/module entry point."""
    args = parse_args(argv)
    result = run_showcase_experiment(seed=args.seed)
    files = write_showcase_artifacts(result, args.output_dir)

    print("=" * 68)
    print("ARC-SCOPE Showcase Experiment")
    print("=" * 68)
    print("This run assembles SCOPE-style inputs from the repo's core stack and")
    print("fits a proxy fluorescence response. It does not execute scope-rtm.")
    print()
    print(f"Timesteps:               {result.summary.n_time_steps}")
    print(f"Peak LAI:                {result.summary.peak_lai:.3f}")
    print(f"Mean direct fraction:    {result.summary.mean_direct_fraction:.3f}")
    print(f"Mean diffuse fraction:   {result.summary.mean_diffuse_fraction:.3f}")
    print(f"True fqe:                {result.summary.true_fqe:.5f}")
    print(f"Initial fqe:             {result.summary.initial_fqe:.5f}")
    print(f"Optimized fqe:           {result.summary.optimized_fqe:.5f}")
    print(f"Initial RMSE:            {result.summary.rmse_initial:.5f}")
    print(f"Optimized RMSE:          {result.summary.rmse_optimized:.5f}")
    print(f"Relative fqe error (%):  {result.summary.relative_fqe_error_pct:.2f}")
    print()
    print("Artifacts written:")
    for name, path in files.items():
        print(f"  {name:>16s}: {path}")


def _build_showcase_mask() -> np.ndarray:
    """Build an irregular field mask to mimic real ARC valid-pixel geometry."""
    ny, nx = 6, 8
    mask = np.zeros((ny, nx), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[1, 0] = True
    mask[1, -1] = True
    mask[-2, 0] = True
    mask[-2, -1] = True
    mask[2, 0] = True
    mask[3, -1] = True
    return mask


def _build_showcase_arc_inputs(
    mask: np.ndarray,
    doys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create seasonal ARC-like arrays with spatial heterogeneity."""
    bounds = load_geojson_bounds(TEST_FIELD_GEOJSON)
    minx, _miny, _maxx, maxy = bounds
    ny, nx = mask.shape
    geotransform = np.array(
        [minx, 0.0005, 0.0, maxy, 0.0, -0.0005],
        dtype=np.float64,
    )

    n_valid = int((~mask).sum())
    nt = int(doys.size)
    valid_positions = np.argwhere(~mask)
    time_phase = np.linspace(0.0, 1.0, nt)

    lai_curve = 0.7 + 4.6 * np.sin(np.pi * time_phase) ** 1.25
    cab_curve = 25.0 + 28.0 * np.sin(np.pi * time_phase) ** 1.1
    cw_curve = 0.010 + 0.015 * np.sin(np.pi * time_phase) ** 1.35
    cm_curve = 0.006 + 0.010 * np.sin(np.pi * time_phase) ** 1.15
    n_curve = 1.35 + 0.28 * np.sin(np.pi * time_phase)
    ala_curve = 63.0 - 10.0 * np.sin(np.pi * time_phase)
    cbrown_curve = 0.10 + 0.55 * (1.0 - np.sin(np.pi * time_phase) ** 1.5)

    post_bio_tensor = np.zeros((n_valid, len(BIO_BANDS), nt), dtype=np.float64)
    scale_data = np.zeros((n_valid, 15), dtype=np.float64)

    for i, (y_idx, x_idx) in enumerate(valid_positions):
        pixel_bias = 1.0 + 0.05 * (x_idx / max(nx - 1, 1)) - 0.04 * (y_idx / max(ny - 1, 1))
        senescence_shift = 1.0 - 0.03 * np.cos((x_idx + y_idx + 1) / 3.0)

        physical_bands = np.vstack(
            [
                n_curve * pixel_bias,
                cab_curve * pixel_bias,
                cm_curve * pixel_bias,
                cw_curve * pixel_bias,
                lai_curve * pixel_bias * senescence_shift,
                ala_curve / pixel_bias,
                cbrown_curve / senescence_shift,
            ]
        )
        for band_index, scale in enumerate(BIO_SCALES):
            post_bio_tensor[i, band_index, :] = np.round(physical_bands[band_index] / scale, 0)

        scale_data[i, :7] = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * pixel_bias
        scale_data[i, 7:11] = np.array([0.15, 0.35, 0.55, 0.85]) + 0.02 * x_idx
        scale_data[i, 11] = 0.24 + 0.015 * x_idx      # BSMBrightness
        scale_data[i, 12] = 16.0 + 1.1 * y_idx        # BSMlat
        scale_data[i, 13] = 28.0 + 2.5 * x_idx        # BSMlon
        scale_data[i, 14] = 22.0 + 2.0 * y_idx        # SMC

    return post_bio_tensor, scale_data, geotransform


def _load_showcase_weather(times: np.ndarray) -> xr.Dataset:
    """Load the bundled showcase weather file through LocalProvider."""
    var_map = {
        "sw_down_wm2": "Rin",
        "lw_down_wm2": "Rli",
        "air_temp_c": "Ta",
        "vapour_pressure_hpa": "ea",
        "pressure_hpa": "p",
        "wind_speed_ms": "u",
    }
    start = pd.Timestamp(times.min()).to_pydatetime()
    end = pd.Timestamp(times.max()).to_pydatetime()
    provider = LocalProvider(
        file_path=SHOWCASE_WEATHER_CSV,
        var_map=var_map,
        time_column="time",
    )
    ds = provider.fetch(
        bounds=load_geojson_bounds(TEST_FIELD_GEOJSON),
        time_range=(start, end),
    )
    ds = ds.sortby("time")
    # Interpolate weather onto observation times (the CSV may not have
    # exact matching dates after extending the coverage window)
    return ds.interp(time=times, method="linear")


def _build_proxy_experiment_dataset(
    *,
    post_bio_da: xr.DataArray,
    weather_ds: xr.Dataset,
    observation_ds: xr.Dataset,
) -> xr.Dataset:
    """Collapse gridded inputs into a time-series dataset for proxy modelling."""
    lai = post_bio_da.sel(band="lai").mean(dim=("y", "x"), skipna=True)
    cab = post_bio_da.sel(band="cab").mean(dim=("y", "x"), skipna=True)
    cw = post_bio_da.sel(band="cw").mean(dim=("y", "x"), skipna=True)

    direct, diffuse = partition_shortwave(
        weather_ds["Rin"].values,
        observation_ds["solar_zenith_angle"].values,
        pd.to_datetime(observation_ds.coords["time"].values).dayofyear.values,
    )
    diffuse_fraction = np.where(weather_ds["Rin"].values > 0.0, diffuse / weather_ds["Rin"].values, 0.0)

    return xr.Dataset(
        {
            "lai": ("time", lai.values),
            "cab": ("time", cab.values),
            "cw": ("time", cw.values),
            "Rin": ("time", weather_ds["Rin"].values),
            "Rli": ("time", weather_ds["Rli"].values),
            "Ta": ("time", weather_ds["Ta"].values),
            "ea": ("time", weather_ds["ea"].values),
            "u": ("time", weather_ds["u"].values),
            "solar_zenith_angle": ("time", observation_ds["solar_zenith_angle"].values),
            "Rin_direct": ("time", direct),
            "Rin_diffuse": ("time", diffuse),
            "diffuse_fraction": ("time", diffuse_fraction),
        },
        coords={"time": observation_ds.coords["time"].values},
    )


def _build_proxy_observations(
    *,
    experiment_ds: xr.Dataset,
    true_fqe: float,
    seed: int,
) -> xr.Dataset:
    """Create deterministic noisy observations from the proxy model."""
    rng = np.random.default_rng(seed)
    truth = _run_proxy_fluorescence_model(
        experiment_ds.assign(fqe=true_fqe)
    )
    noise = rng.normal(loc=0.0, scale=0.004, size=truth.sizes["time"])
    observed = np.clip(truth["proxy_sif"].values + noise, a_min=0.0, a_max=None)
    return xr.Dataset(
        {"proxy_sif": ("time", observed)},
        coords={"time": truth.coords["time"].values},
    )


def _run_proxy_fluorescence_model(dataset: xr.Dataset) -> xr.Dataset:
    """Simple proxy canopy model used for the showcase calibration.

    The output is intentionally labelled ``proxy_sif`` to avoid confusion with
    actual SCOPE fluorescence products.
    """
    lai = dataset["lai"]
    cab = dataset["cab"]
    cw = dataset["cw"]
    rin_direct = dataset["Rin_direct"]
    rin_diffuse = dataset["Rin_diffuse"]
    sza = np.deg2rad(dataset["solar_zenith_angle"])
    ta = dataset["Ta"]
    fqe = dataset["fqe"]

    canopy_absorption = (1.0 - np.exp(-0.72 * lai)) * np.clip(cab / 58.0, 0.25, 1.4)
    water_modifier = np.clip(0.72 + 14.0 * cw, 0.72, 1.08)
    diffuse_gain = 0.82 + 0.28 * dataset["diffuse_fraction"]
    heat_penalty = np.clip(1.0 - 0.02 * np.maximum(ta - 26.0, 0.0), 0.68, 1.0)
    illumination = (0.58 * rin_direct + 0.92 * rin_diffuse) * np.clip(np.cos(sza), 0.2, 1.0)
    proxy_sif = fqe * canopy_absorption * water_modifier * diffuse_gain * heat_penalty * illumination / 30.0

    return xr.Dataset(
        {"proxy_sif": ("time", np.asarray(proxy_sif.values, dtype=np.float64))},
        coords={"time": dataset.coords["time"].values},
    )


def _build_timeseries_frame(
    *,
    experiment_ds: xr.Dataset,
    observations_ds: xr.Dataset,
    initial_output_ds: xr.Dataset,
    fitted_output_ds: xr.Dataset,
) -> pd.DataFrame:
    """Build a flat table for reporting and docs artifacts."""
    times = pd.to_datetime(experiment_ds.coords["time"].values)
    return pd.DataFrame(
        {
            "date": times.strftime("%Y-%m-%d"),
            "lai": np.round(experiment_ds["lai"].values, 4),
            "cab": np.round(experiment_ds["cab"].values, 4),
            "cw": np.round(experiment_ds["cw"].values, 5),
            "Rin": np.round(experiment_ds["Rin"].values, 3),
            "Rin_direct": np.round(experiment_ds["Rin_direct"].values, 3),
            "Rin_diffuse": np.round(experiment_ds["Rin_diffuse"].values, 3),
            "direct_fraction": np.round(1.0 - experiment_ds["diffuse_fraction"].values, 4),
            "diffuse_fraction": np.round(experiment_ds["diffuse_fraction"].values, 4),
            "proxy_sif_observed": np.round(observations_ds["proxy_sif"].values, 5),
            "proxy_sif_initial": np.round(initial_output_ds["proxy_sif"].values, 5),
            "proxy_sif_fitted": np.round(fitted_output_ds["proxy_sif"].values, 5),
        }
    )


def _write_line_chart_svg(
    *,
    path: Path,
    x_values: Iterable[str],
    series: Mapping[str, Iterable[float]],
    colors: Mapping[str, str],
    title: str,
    y_label: str,
) -> None:
    """Write a small self-contained SVG line chart without plotting deps."""
    x_list = list(x_values)
    series_data = {name: [float(v) for v in values] for name, values in series.items()}
    all_values = [value for values in series_data.values() for value in values]

    width = 920
    height = 360
    margin_left = 72
    margin_right = 28
    margin_top = 42
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    y_min = min(all_values)
    y_max = max(all_values)
    if np.isclose(y_min, y_max):
        y_max = y_min + 1.0
    y_pad = 0.08 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    def x_coord(index: int) -> float:
        if len(x_list) == 1:
            return margin_left + plot_width / 2.0
        return margin_left + index * plot_width / (len(x_list) - 1)

    def y_coord(value: float) -> float:
        scaled = (value - y_min) / (y_max - y_min)
        return margin_top + plot_height - scaled * plot_height

    ticks = np.linspace(y_min, y_max, 5)
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>'
        'text{font-family:Arial,sans-serif;fill:#111827;}'
        '.axis{stroke:#9ca3af;stroke-width:1;}'
        '.grid{stroke:#e5e7eb;stroke-width:1;}'
        '.legend{font-size:12px;}'
        '.tick{font-size:11px;fill:#4b5563;}'
        '.title{font-size:18px;font-weight:700;}'
        '</style>',
        f'<text class="title" x="{margin_left}" y="28">{title}</text>',
        f'<text class="tick" x="{margin_left}" y="{height - 16}">Time step</text>',
        (
            f'<text class="tick" transform="translate(20 {margin_top + plot_height / 2}) rotate(-90)">'
            f"{y_label}</text>"
        ),
    ]

    for tick in ticks:
        y = y_coord(float(tick))
        lines.append(f'<line class="grid" x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" />')
        lines.append(f'<text class="tick" x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end">{tick:.2f}</text>')

    lines.append(
        f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" />'
    )
    lines.append(
        f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" />'
    )

    for idx, label in enumerate(x_list):
        x = x_coord(idx)
        lines.append(
            f'<line class="grid" x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_height}" />'
        )
        lines.append(
            f'<text class="tick" x="{x:.2f}" y="{height - 42}" text-anchor="middle">{label}</text>'
        )

    legend_x = width - margin_right - 210
    legend_y = 24
    for idx, (name, values) in enumerate(series_data.items()):
        points = " ".join(
            f"{x_coord(i):.2f},{y_coord(value):.2f}"
            for i, value in enumerate(values)
        )
        color = colors.get(name, "#111827")
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}" />'
        )
        for i, value in enumerate(values):
            lines.append(
                f'<circle cx="{x_coord(i):.2f}" cy="{y_coord(value):.2f}" r="3.5" fill="{color}" />'
            )
        legend_top = legend_y + idx * 18
        lines.append(
            f'<line x1="{legend_x}" y1="{legend_top}" x2="{legend_x + 20}" y2="{legend_top}" stroke="{color}" stroke-width="3" />'
        )
        lines.append(
            f'<text class="legend" x="{legend_x + 28}" y="{legend_top + 4}">{name}</text>'
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_showcase_dashboard_html(
    result: ShowcaseExperimentResult,
    path: Path,
) -> None:
    """Write a single-scroll interactive Plotly dashboard.

    The layout shows the full causal chain visible at once:
    KPIs → combined overview → canopy inputs → radiation forcing →
    calibration result → fit diagnostics. No hidden tabs.
    """
    ts = result.timeseries
    s = result.summary

    def _j(col: str) -> str:
        return json.dumps(ts[col].tolist())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ARC-SCOPE Showcase</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{ font-family:Inter,system-ui,-apple-system,sans-serif;
  --bg:#f8fafc;--panel:#fff;--border:#e2e8f0;--text:#0f172a;--muted:#64748b;--accent:#0f766e; }}
*{{ box-sizing:border-box; margin:0; }}
body{{ background:var(--bg); color:var(--text); padding:1rem; }}
.shell{{ max-width:1280px; margin:0 auto; display:grid; gap:0.85rem; }}
.panel{{ background:var(--panel); border:1px solid var(--border); border-radius:14px;
  padding:1rem; box-shadow:0 1px 3px rgba(0,0,0,0.04); }}
h1{{ font-size:1.35rem; font-weight:700; margin:0; }}
h2{{ font-size:0.95rem; font-weight:600; color:var(--muted); text-transform:uppercase;
  letter-spacing:0.04em; margin:0 0 0.4rem; }}
.sub{{ color:var(--muted); font-size:0.85rem; line-height:1.5; }}
.kpis{{ display:grid; grid-template-columns:repeat(auto-fill,minmax(115px,1fr)); gap:0.5rem; margin-top:0.6rem; }}
.kpi{{ text-align:center; padding:0.5rem 0.3rem; border:1px solid var(--border); border-radius:10px; }}
.kv{{ font-size:1.25rem; font-weight:700; color:var(--accent); }}
.kl{{ font-size:0.68rem; color:var(--muted); margin-top:0.1rem; }}
.row{{ display:grid; gap:0.85rem; }}
.c2{{ grid-template-columns:1fr 1fr; }}
.c3{{ grid-template-columns:1fr 1fr 1fr; }}
.plot{{ min-height:320px; }}
.plot-lg{{ min-height:400px; }}
.note{{ color:var(--muted); font-size:0.8rem; line-height:1.5; }}
.arrow{{ text-align:center; font-size:1.6rem; color:var(--border); padding:0.2rem 0; }}
@media(max-width:860px){{ .c2,.c3{{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="shell">

  <!-- Header + KPIs -->
  <div class="panel">
    <h1>ARC-SCOPE Showcase</h1>
    <p class="sub">Synthetic ARC retrieval &rarr; local weather &rarr; radiation partitioning
    &rarr; proxy SIF model &rarr; parameter optimisation. All plots visible &mdash; scroll to follow the causal chain.</p>
    <div class="kpis">
      <div class="kpi"><div class="kv">{s.n_time_steps}</div><div class="kl">Timesteps</div></div>
      <div class="kpi"><div class="kv">{s.peak_lai:.2f}</div><div class="kl">Peak LAI</div></div>
      <div class="kpi"><div class="kv">{s.mean_direct_fraction:.1%}</div><div class="kl">Mean direct frac</div></div>
      <div class="kpi"><div class="kv">{s.true_fqe:.4f}</div><div class="kl">True fqe</div></div>
      <div class="kpi"><div class="kv">{s.optimized_fqe:.4f}</div><div class="kl">Fitted fqe</div></div>
      <div class="kpi"><div class="kv">{s.relative_fqe_error_pct:.1f}%</div><div class="kl">fqe error</div></div>
      <div class="kpi"><div class="kv">{s.rmse_initial:.4f}</div><div class="kl">Initial RMSE</div></div>
      <div class="kpi"><div class="kv">{s.rmse_optimized:.4f}</div><div class="kl">Final RMSE</div></div>
    </div>
  </div>

  <!-- 1: Combined overview — LAI + Rin + SIF on shared x-axis -->
  <div class="panel">
    <h2>1 &middot; Season overview</h2>
    <p class="note">LAI drives canopy absorption, Rin drives illumination, and their product
    (scaled by fqe) controls the proxy SIF signal. The three curves share the same time axis
    so the seasonal coupling is immediately visible.</p>
    <div id="pOverview" class="plot-lg"></div>
  </div>

  <!-- 2: Canopy state -->
  <div class="panel"><h2>2 &middot; Canopy biophysical inputs</h2></div>
  <div class="row c3">
    <div class="panel"><div id="pLAI" class="plot"></div></div>
    <div class="panel"><div id="pCab" class="plot"></div></div>
    <div class="panel"><div id="pCw" class="plot"></div></div>
  </div>

  <div class="arrow">&darr;</div>

  <!-- 3: Radiation -->
  <div class="panel"><h2>3 &middot; Radiation partitioning</h2>
    <p class="note">Total shortwave (Rin) is split into direct beam and diffuse sky components
    using the BRL clearness-index model. The diffuse fraction controls how much light
    penetrates deep into the canopy.</p>
  </div>
  <div class="row c2">
    <div class="panel"><div id="pRad" class="plot"></div></div>
    <div class="panel"><div id="pFrac" class="plot"></div></div>
  </div>

  <div class="arrow">&darr;</div>

  <!-- 4: Calibration -->
  <div class="panel"><h2>4 &middot; Proxy SIF calibration</h2>
    <p class="note">The proxy model generates a SIF-like signal from the canopy and radiation
    inputs. The optimiser adjusts <code>fqe</code> (fluorescence quantum efficiency) from
    {s.initial_fqe:.4f} to {s.optimized_fqe:.4f} to match the noisy synthetic observations,
    reducing RMSE from {s.rmse_initial:.4f} to {s.rmse_optimized:.5f}.</p>
  </div>
  <div class="panel"><div id="pSIF" class="plot-lg"></div></div>

  <div class="arrow">&darr;</div>

  <!-- 5: Diagnostics -->
  <div class="panel"><h2>5 &middot; Fit diagnostics</h2></div>
  <div class="row c2">
    <div class="panel"><div id="pResid" class="plot"></div></div>
    <div class="panel"><div id="pScatter" class="plot"></div></div>
  </div>
</div>

<script>
const D={_j("date")};
const LAI={_j("lai")},Cab={_j("cab")},Cw={_j("cw")};
const Rin={_j("Rin")},Dir={_j("Rin_direct")},Dif={_j("Rin_diffuse")};
const Obs={_j("proxy_sif_observed")},Init={_j("proxy_sif_initial")},Fit={_j("proxy_sif_fitted")};
const M={{t:36,l:52,r:18,b:36}},C={{responsive:true,displayModeBar:true,modeBarButtonsToRemove:["lasso2d","select2d"]}};
const frac=Dir.map((d,i)=>Rin[i]>0?d/Rin[i]:0);
const residI=Obs.map((o,i)=>o-Init[i]),residF=Obs.map((o,i)=>o-Fit[i]);
const laiNorm=LAI.map(v=>v/Math.max(...LAI)),rinNorm=Rin.map(v=>v/Math.max(...Rin)),
      sifNorm=Obs.map(v=>v/Math.max(...Obs));

/* 1: Overview — normalised overlay */
Plotly.newPlot('pOverview',[
  {{x:D,y:laiNorm,mode:'lines+markers',name:'LAI (norm)',line:{{color:'#15803d',width:2.5}},marker:{{size:5}}}},
  {{x:D,y:rinNorm,mode:'lines+markers',name:'Rin (norm)',line:{{color:'#ea580c',width:2.5}},marker:{{size:5}}}},
  {{x:D,y:sifNorm,mode:'lines+markers',name:'Proxy SIF (norm)',line:{{color:'#7c3aed',width:3}},marker:{{size:7}}}},
],{{title:'Normalised season overview: canopy &times; radiation &rarr; SIF',
  margin:{{...M,t:42}},xaxis:{{title:'Date'}},yaxis:{{title:'Normalised (0-1)',range:[-0.05,1.1]}},
  legend:{{orientation:'h',y:1.12}}}},C);

/* 2: Canopy — individual panels */
Plotly.newPlot('pLAI',[{{x:D,y:LAI,mode:'lines+markers',line:{{color:'#15803d',width:2.5}},
  marker:{{size:7}},fill:'tozeroy',fillcolor:'rgba(21,128,61,0.08)'}}],
  {{title:'LAI',margin:M,yaxis:{{title:'m\u00b2/m\u00b2'}}}},C);
Plotly.newPlot('pCab',[{{x:D,y:Cab,mode:'lines+markers',line:{{color:'#2563eb',width:2.5}},
  marker:{{size:7}},fill:'tozeroy',fillcolor:'rgba(37,99,235,0.08)'}}],
  {{title:'Chlorophyll a+b',margin:M,yaxis:{{title:'\u00b5g/cm\u00b2'}}}},C);
Plotly.newPlot('pCw',[{{x:D,y:Cw,mode:'lines+markers',line:{{color:'#0891b2',width:2.5}},
  marker:{{size:7}},fill:'tozeroy',fillcolor:'rgba(8,145,178,0.08)'}}],
  {{title:'Leaf water content',margin:M,yaxis:{{title:'g/cm\u00b2'}}}},C);

/* 3: Radiation */
Plotly.newPlot('pRad',[
  {{x:D,y:Rin,mode:'lines+markers',name:'Total Rin',line:{{color:'#475569',width:1.5,dash:'dot'}},marker:{{size:4}}}},
  {{x:D,y:Dir,mode:'lines+markers',name:'Direct',line:{{color:'#c2410c',width:2.5}},marker:{{size:6}},
    fill:'tozeroy',fillcolor:'rgba(194,65,12,0.12)'}},
  {{x:D,y:Dif,mode:'lines+markers',name:'Diffuse',line:{{color:'#2563eb',width:2.5}},marker:{{size:6}},
    fill:'tozeroy',fillcolor:'rgba(37,99,235,0.08)'}},
],{{title:'Shortwave radiation',margin:M,yaxis:{{title:'W/m\u00b2'}},legend:{{orientation:'h',y:1.08}}}},C);

Plotly.newPlot('pFrac',[
  {{x:D,y:frac,mode:'lines+markers',name:'Direct / Total',line:{{color:'#7c3aed',width:2.5}},
    marker:{{size:7}},fill:'tozeroy',fillcolor:'rgba(124,58,237,0.08)'}},
],{{title:'Direct fraction',margin:M,yaxis:{{title:'Fraction',range:[0,0.35]}}}},C);

/* 4: Calibration */
Plotly.newPlot('pSIF',[
  {{x:D,y:Obs,mode:'markers',name:'Observed (noisy)',marker:{{size:11,color:'#111827',symbol:'circle-open',line:{{width:2.5}}}}}},
  {{x:D,y:Init,mode:'lines+markers',name:'Initial (fqe={s.initial_fqe:.4f})',
    line:{{color:'#dc2626',width:2,dash:'dash'}},marker:{{size:5}}}},
  {{x:D,y:Fit,mode:'lines+markers',name:'Optimised (fqe={s.optimized_fqe:.4f})',
    line:{{color:'#16a34a',width:3}},marker:{{size:7}}}},
],{{title:'Proxy SIF: observed vs modelled',margin:{{...M,t:42}},
  xaxis:{{title:'Date'}},yaxis:{{title:'Proxy SIF (a.u.)'}},
  legend:{{orientation:'h',y:1.12}}}},C);

/* 5: Diagnostics */
Plotly.newPlot('pResid',[
  {{x:D,y:residI,mode:'lines+markers',name:'Initial',line:{{color:'#dc2626',width:2}},marker:{{size:5}}}},
  {{x:D,y:residF,mode:'lines+markers',name:'Optimised',line:{{color:'#16a34a',width:2.5}},marker:{{size:6}}}},
  {{x:D,y:D.map(()=>0),mode:'lines',line:{{color:'#94a3b8',dash:'dash',width:1}},showlegend:false}},
],{{title:'Residuals (obs \u2212 model)',margin:M,yaxis:{{title:'Residual',zeroline:true}},
  legend:{{orientation:'h',y:1.08}}}},C);

const lo=Math.min(...Obs,...Init,...Fit)*0.9, hi=Math.max(...Obs,...Init,...Fit)*1.1;
Plotly.newPlot('pScatter',[
  {{x:Obs,y:Fit,mode:'markers',name:'Optimised',
    marker:{{size:10,color:'#16a34a',line:{{width:1,color:'#15803d'}}}},
    hovertemplate:'obs=%{{x:.4f}}<br>fit=%{{y:.4f}}<extra></extra>'}},
  {{x:Obs,y:Init,mode:'markers',name:'Initial',
    marker:{{size:8,color:'#dc2626',symbol:'x',line:{{width:1.5}}}},
    hovertemplate:'obs=%{{x:.4f}}<br>init=%{{y:.4f}}<extra></extra>'}},
  {{x:[lo,hi],y:[lo,hi],mode:'lines',line:{{color:'#94a3b8',dash:'dash',width:1}},showlegend:false}},
],{{title:'Observed vs modelled (1:1 line)',margin:M,
  xaxis:{{title:'Observed',range:[lo,hi]}},yaxis:{{title:'Modelled',range:[lo,hi],scaleanchor:'x'}},
  legend:{{orientation:'h',y:1.08}}}},C);
</script>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
