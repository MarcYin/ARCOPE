"""Individual pipeline step functions.

These are composable building blocks for users who want to run partial
pipelines or customise the data flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from arc_scope.bridge.convert import arc_arrays_to_scope_inputs, arc_npz_to_scope_inputs
from arc_scope.pipeline.config import PipelineConfig
from arc_scope.utils.geometry import solar_position
from arc_scope.utils.io import load_geojson_bounds, load_geojson_centroid
from arc_scope.utils.types import PathLike


@dataclass
class ArcResult:
    """Container for ARC retrieval outputs."""

    scale_data: np.ndarray
    post_bio_tensor: np.ndarray
    post_bio_unc_tensor: np.ndarray
    mask: np.ndarray
    doys: np.ndarray
    geotransform: np.ndarray | None = None
    crs: Any = None


def retrieve_arc(config: PipelineConfig) -> ArcResult:
    """Run ARC biophysical parameter retrieval.

    Parameters
    ----------
    config:
        Pipeline configuration with field definition and ARC options.

    Returns
    -------
    ArcResult containing the retrieved biophysical parameters.
    """
    try:
        from arc import arc_field
    except ImportError:
        raise ImportError(
            "ARC is required for retrieval. Install with: pip install arc-scope[arc]"
        )

    s2_folder = str(config.s2_data_folder or Path("./S2_data"))
    output_path = str(config.output_dir / "arc_output.npz")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    scale_data, post_bio_tensor, post_bio_unc_tensor, mask, doys = arc_field(
        s2_start_date=config.start_date,
        s2_end_date=config.end_date,
        geojson_path=str(config.geojson_path),
        start_of_season=config.start_of_season,
        crop_type=config.crop_type,
        output_file_path=output_path,
        num_samples=config.num_samples,
        growth_season_length=config.growth_season_length,
        S2_data_folder=s2_folder,
        data_source=config.data_source,
    )

    # Recover geotransform and CRS from the saved NPZ
    geotransform = None
    crs = None
    npz_path = Path(output_path)
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=True) as payload:
            geotransform = np.asarray(payload["geotransform"], dtype=np.float64)
            crs = payload.get("crs")
            if isinstance(crs, np.ndarray):
                crs = crs.item()
                if isinstance(crs, bytes):
                    crs = crs.decode("utf-8")

    return ArcResult(
        scale_data=scale_data,
        post_bio_tensor=post_bio_tensor,
        post_bio_unc_tensor=post_bio_unc_tensor,
        mask=mask,
        doys=doys,
        geotransform=geotransform,
        crs=crs,
    )


def bridge_arc_to_scope(
    arc_result: ArcResult,
    year: int,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Convert ARC retrieval outputs to SCOPE-compatible DataArrays.

    Parameters
    ----------
    arc_result:
        Output from :func:`retrieve_arc`.
    year:
        Calendar year for datetime coordinates.

    Returns
    -------
    Tuple of ``(post_bio_da, post_bio_scale_da)``.
    """
    if arc_result.geotransform is None:
        raise ValueError("ArcResult must include geotransform for SCOPE conversion")

    return arc_arrays_to_scope_inputs(
        post_bio_tensor=arc_result.post_bio_tensor,
        scale_data=arc_result.scale_data,
        mask=arc_result.mask,
        doys=arc_result.doys,
        geotransform=arc_result.geotransform,
        crs=arc_result.crs,
        year=year,
    )


def build_observation_dataset(
    doys: np.ndarray,
    year: int,
    geojson_path: PathLike,
    *,
    viewing_zenith: float = 0.0,
    viewing_azimuth: float = 0.0,
    overpass_hour: float = 10.5,
    duplicate_step_minutes: float = 5.0,
) -> xr.Dataset:
    """Build an observation geometry dataset for SCOPE.

    Computes solar angles from field location and overpass time, and uses
    the provided (or default nadir) viewing geometry. Repeated acquisitions on
    the same day are offset by a small, deterministic minute increment so the
    time index remains unique for SCOPE input preparation.

    Parameters
    ----------
    doys:
        Day-of-year array for observation times.
    year:
        Calendar year.
    geojson_path:
        Path to field GeoJSON for determining location.
    viewing_zenith:
        Sensor viewing zenith angle in degrees (default nadir = 0).
    viewing_azimuth:
        Sensor viewing azimuth angle in degrees.
    overpass_hour:
        Local solar time of satellite overpass (default 10:30 for S2).
    duplicate_step_minutes:
        Minute offset applied to repeated same-day acquisitions.

    Returns
    -------
    xr.Dataset with variables for solar/viewing geometry and a ``delta_time``
    variable for SCOPE's observation time grid.
    """
    lon, lat = load_geojson_centroid(geojson_path)

    times = []
    szas = []
    saas = []
    seen_per_day: dict[int, int] = {}
    for doy in doys:
        day = int(doy)
        duplicate_index = seen_per_day.get(day, 0)
        seen_per_day[day] = duplicate_index + 1

        dt = datetime(year, 1, 1) + timedelta(
            days=day - 1,
            hours=overpass_hour,
            minutes=duplicate_index * duplicate_step_minutes,
        )
        sza, saa = solar_position(lat, lon, dt)
        times.append(np.datetime64(dt))
        szas.append(float(sza))
        saas.append(float(saa))

    time_coords = np.array(times, dtype="datetime64[ns]")

    ds = xr.Dataset(
        {
            "solar_zenith_angle": ("time", np.array(szas)),
            "viewing_zenith_angle": ("time", np.full(len(doys), viewing_zenith)),
            "solar_azimuth_angle": ("time", np.array(saas)),
            "viewing_azimuth_angle": ("time", np.full(len(doys), viewing_azimuth)),
            "delta_time": ("time", time_coords),
        },
        coords={"time": time_coords},
    )
    return ds


def fetch_weather(
    config: PipelineConfig,
    time_range: tuple[datetime, datetime] | None = None,
) -> xr.Dataset:
    """Fetch weather data using the configured provider.

    Parameters
    ----------
    config:
        Pipeline configuration with weather provider settings.
    time_range:
        Override time range. Defaults to ``(start_date, end_date)``.

    Returns
    -------
    xr.Dataset with SCOPE-compatible weather variables.
    """
    from arc_scope.utils.io import load_geojson_bounds

    bounds = load_geojson_bounds(config.geojson_path)
    if time_range is None:
        time_range = (
            datetime.strptime(config.start_date, "%Y-%m-%d"),
            datetime.strptime(config.end_date, "%Y-%m-%d"),
        )

    if config.weather_provider == "era5":
        from arc_scope.weather.era5 import ERA5Provider

        provider = ERA5Provider(**config.weather_config)
    elif config.weather_provider == "local":
        from arc_scope.weather.local import LocalProvider

        provider = LocalProvider(**config.weather_config)
    else:
        raise ValueError(f"Unknown weather provider: {config.weather_provider}")

    return provider.fetch(bounds, time_range)


def prepare_scope_dataset(
    post_bio_da: xr.DataArray,
    post_bio_scale_da: xr.DataArray,
    weather_ds: xr.Dataset,
    observation_ds: xr.Dataset,
    config: PipelineConfig,
) -> xr.Dataset:
    """Prepare a runner-ready SCOPE dataset from all inputs.

    Parameters
    ----------
    post_bio_da, post_bio_scale_da:
        Bridge outputs from :func:`bridge_arc_to_scope`.
    weather_ds:
        Weather data from :func:`fetch_weather`.
    observation_ds:
        Observation geometry from :func:`build_observation_dataset`.
    config:
        Pipeline configuration.

    Returns
    -------
    Runner-ready xr.Dataset for SCOPE.
    """
    try:
        from scope.io.prepare import prepare_scope_input_dataset

        return prepare_scope_input_dataset(
            weather_ds=weather_ds,
            observation_ds=observation_ds,
            post_bio_da=post_bio_da,
            post_bio_scale_da=post_bio_scale_da,
            scope_root_path=config.scope_root_path,
            scope_options=config.resolved_scope_options,
        )
    except ImportError:
        raise ImportError(
            "SCOPE is required for simulation. Install with: pip install arc-scope[scope]"
        )


def run_scope_simulation(
    scope_dataset: xr.Dataset,
    config: PipelineConfig,
) -> xr.Dataset:
    """Execute the SCOPE simulation.

    Parameters
    ----------
    scope_dataset:
        Prepared dataset from :func:`prepare_scope_dataset`.
    config:
        Pipeline configuration with SCOPE options.

    Returns
    -------
    xr.Dataset with SCOPE simulation outputs.
    """
    try:
        import torch
        from scope import SimulationConfig, ScopeGridRunner, campbell_lidf
        from scope.data import ScopeGridDataModule
        from scope.spectral.fluspect import FluspectModel
    except ImportError:
        raise ImportError(
            "SCOPE and PyTorch are required. Install with: pip install arc-scope[scope]"
        )

    _patch_scope_fluspect_stacked_layers(FluspectModel, torch)

    # Determine time bounds from dataset
    times = scope_dataset.coords["time"].values
    start_time = pd.Timestamp(times.min())
    end_time = pd.Timestamp(times.max())

    # Build spatial bounds from dataset coordinates
    x = scope_dataset.coords["x"].values
    y = scope_dataset.coords["y"].values
    roi_bounds = (float(x.min()), float(y.min()), float(x.max()), float(y.max()))

    dtype_map = {"float32": torch.float32, "float64": torch.float64}
    torch_dtype = dtype_map.get(config.dtype, torch.float64)
    device = torch.device(config.device)

    sim_config = SimulationConfig(
        roi_bounds=roi_bounds,
        start_time=start_time,
        end_time=end_time,
        device=str(device),
        dtype=torch_dtype,
    )

    # Build leaf inclination distribution (Campbell spherical, 57 deg)
    lidf = campbell_lidf(57.0, device=device, dtype=torch_dtype)

    runner = ScopeGridRunner.from_scope_assets(
        lidf=lidf,
        device=device,
        dtype=torch_dtype,
        scope_root_path=config.scope_root_path,
    )

    # Build data module with required variables
    required_vars = list(scope_dataset.data_vars)
    data_module = ScopeGridDataModule(scope_dataset, sim_config, required_vars=required_vars)

    # Build variable mapping (identity for the prepared dataset)
    varmap = {v: v for v in scope_dataset.data_vars}

    return runner.run_scope_dataset(
        data_module,
        varmap=varmap,
        scope_options=config.resolved_scope_options,
    )


def _patch_scope_fluspect_stacked_layers(fluspect_model_cls: type, torch_module: Any) -> None:
    """Patch scope-rtm 0.2.0 to broadcast ``N`` before boolean masking."""
    if getattr(fluspect_model_cls._stacked_layers, "_arc_scope_patched", False):
        return

    def _stacked_layers(self, r, t, N):
        D = torch_module.sqrt(
            torch_module.clamp(
                (1 + r + t) * (1 + r - t) * (1 - r + t) * (1 - r - t),
                min=0.0,
            )
        )
        rq = r**2
        tq = t**2
        a = (1 + rq - tq + D) / (2 * r.clamp(min=1e-9))
        b = (1 - rq + tq + D) / (2 * t.clamp(min=1e-9))
        N_layers = N.unsqueeze(-1).expand_as(t)
        bNm1 = b ** (N_layers - 1)
        bN2 = bNm1**2
        a2 = a**2
        denom = a2 * bN2 - 1
        Rsub = a * (bN2 - 1) / denom
        Tsub = bNm1 * (a2 - 1) / denom
        zero_abs = (r + t) >= 1
        if zero_abs.any():
            denom_zero = t[zero_abs] + (1 - t[zero_abs]) * (N_layers[zero_abs] - 1)
            Tsub[zero_abs] = t[zero_abs] / denom_zero
            Rsub[zero_abs] = 1 - Tsub[zero_abs]
        return Rsub, Tsub

    _stacked_layers._arc_scope_patched = True
    fluspect_model_cls._stacked_layers = _stacked_layers
