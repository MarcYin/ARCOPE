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
from arc_scope.weather.radiation import build_scope_spectral_forcing
from arc_scope.utils.geometry import solar_position
from arc_scope.utils.io import load_geojson_bounds, load_geojson_centroid
from arc_scope.utils.types import PathLike

ENERGY_BALANCE_REQUIRED_VARS = (
    "Cab",
    "Cw",
    "Cdm",
    "LAI",
    "tts",
    "tto",
    "psi",
    "Ta",
    "ea",
    "Ca",
    "Oa",
    "p",
    "z",
    "u",
    "Cd",
    "rwc",
    "z0m",
    "d",
    "h",
    "rss",
    "rbs",
    "Esun_sw",
    "Esky_sw",
    "Vcmax25",
    "BallBerrySlope",
)
ENERGY_BALANCE_FLUORESCENCE_VARS = ("fqe", "Esun_", "Esky_")
ENERGY_BALANCE_DEFAULTS = {
    "Ca": 410.0,
    "Oa": 209.0,
    "Cd": 0.3,
    "rwc": 0.0,
    "rbs": 10.0,
    "Vcmax25": 60.0,
    "BallBerrySlope": 8.0,
}
CROP_HEIGHT_CAP_M = {
    "wheat": 1.2,
    "barley": 1.1,
    "soybean": 1.1,
    "potato": 0.8,
    "rice": 1.3,
    "maize": 2.5,
    "corn": 2.5,
}


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

        dataset = prepare_scope_input_dataset(
            weather_ds=weather_ds,
            observation_ds=observation_ds,
            post_bio_da=post_bio_da,
            post_bio_scale_da=post_bio_scale_da,
            scope_root_path=config.scope_root_path,
            scope_options=config.resolved_scope_options,
        )
        return _augment_scope_dataset(dataset, config)
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
        from scope.io import validate_scope_dataset
        from scope.spectral.fluspect import FluspectModel
    except ImportError:
        raise ImportError(
            "SCOPE and PyTorch are required. Install with: pip install arc-scope[scope]"
        )

    _patch_scope_fluspect_stacked_layers(FluspectModel, torch)
    runner_dataset, spatial_valid = _prepare_runner_dataset(scope_dataset)

    # Determine time bounds from dataset
    times = runner_dataset.coords["time"].values
    start_time = pd.Timestamp(times.min())
    end_time = pd.Timestamp(times.max())

    # Build spatial bounds from dataset coordinates
    x = runner_dataset.coords["x"].values
    y = runner_dataset.coords["y"].values
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
    required_vars = list(runner_dataset.data_vars)
    data_module = ScopeGridDataModule(runner_dataset, sim_config, required_vars=required_vars)

    # Build variable mapping (identity for the prepared dataset)
    varmap = {v: v for v in runner_dataset.data_vars}

    if config.scope_workflow == "energy-balance":
        output = _run_coupled_energy_balance(
            runner,
            data_module,
            varmap=varmap,
            scope_dataset=runner_dataset,
            validate_scope_dataset=validate_scope_dataset,
            soil_heat_method=int(config.resolved_scope_options.get("soil_heat_method", 2)),
        )
        return _apply_spatial_mask(output, spatial_valid)

    validate_scope_dataset(
        runner_dataset,
        workflow="scope",
        scope_options=config.resolved_scope_options,
    )
    output = runner.run_scope_dataset(
        data_module,
        varmap=varmap,
        scope_options=config.resolved_scope_options,
    )
    return _apply_spatial_mask(output, spatial_valid)


def _run_coupled_energy_balance(
    runner: Any,
    data_module: Any,
    *,
    varmap: dict[str, str],
    scope_dataset: xr.Dataset,
    validate_scope_dataset: Any,
    soil_heat_method: int,
) -> xr.Dataset:
    """Execute the explicit coupled energy-balance workflows and merge outputs.

    ``scope-rtm`` exposes coupled fluorescence and coupled thermal as two
    separate entry points. ARC-SCOPE's ``energy-balance`` workflow wraps both
    so users can inspect the combined SIF, thermal, and flux outputs from one
    prepared dataset.
    """
    validate_scope_dataset(scope_dataset, workflow="energy-balance-fluorescence")
    validate_scope_dataset(scope_dataset, workflow="energy-balance-thermal")
    _validate_hidden_energy_balance_inputs(scope_dataset)

    fluorescence_output = runner.run_energy_balance_fluorescence_dataset(
        data_module,
        varmap=varmap,
        soil_heat_method=soil_heat_method,
    )
    thermal_output = runner.run_energy_balance_thermal_dataset(
        data_module,
        varmap=varmap,
        soil_heat_method=soil_heat_method,
    )

    combined = thermal_output.copy()
    for name, data_array in fluorescence_output.data_vars.items():
        if name not in combined.data_vars:
            combined[name] = data_array

    combined.attrs.update(fluorescence_output.attrs)
    combined.attrs.update(thermal_output.attrs)
    combined.attrs["scope_product"] = "energy_balance"
    combined.attrs["scope_components"] = "energy,physiology,fluorescence,thermal"
    combined.attrs["arc_scope_energy_balance"] = (
        "Merged coupled fluorescence and thermal branches from scope-rtm 0.2.x"
    )
    return combined


def _prepare_runner_dataset(scope_dataset: xr.Dataset) -> tuple[xr.Dataset, xr.DataArray | None]:
    """Fill off-field gaps for simulation while preserving the real mask.

    ``scope-rtm`` stacks every y/x/time cell into dense batches and converts
    NaNs to zero internally. That can break workflows which require strictly
    positive leafbio state, especially coupled fluorescence. ARC-SCOPE keeps
    the authored dataset unchanged, but uses a filled copy for execution and
    reapplies the original spatial validity mask to outputs afterwards.
    """
    spatial_valid = _spatial_valid_mask(scope_dataset)
    if spatial_valid is None:
        return scope_dataset, None

    runner_dataset = scope_dataset.copy(deep=True)
    for name, data_array in runner_dataset.data_vars.items():
        if not _is_numeric_dataarray(data_array):
            continue
        if not {"y", "x"}.issubset(data_array.dims):
            continue
        fill_value = _fill_value_for_runner(name, data_array)
        if fill_value is None:
            continue
        runner_dataset[name] = data_array.where(np.isfinite(data_array), other=fill_value)
    return runner_dataset, spatial_valid


def _spatial_valid_mask(scope_dataset: xr.Dataset) -> xr.DataArray | None:
    """Infer the field footprint from finite biophysical state."""
    for name in ("LAI", "Cab", "Cw", "Cdm"):
        if name not in scope_dataset:
            continue
        data_array = scope_dataset[name]
        if not {"y", "x"}.issubset(data_array.dims):
            continue
        reduce_dims = [dim for dim in data_array.dims if dim not in {"y", "x"}]
        mask = np.isfinite(data_array)
        if reduce_dims:
            mask = mask.any(dim=reduce_dims)
        if bool(mask.any().item()):
            return mask.astype(bool)
    return None


def _fill_value_for_runner(name: str, data_array: xr.DataArray) -> float | None:
    """Choose a finite placeholder for masked-out pixels during execution."""
    finite = data_array.where(np.isfinite(data_array))
    if finite.count().item() == 0:
        return 0.01 if name == "fqe" else 0.0

    median_value = float(finite.median(skipna=True).item())
    if name == "fqe":
        return max(median_value, 0.01)
    if name in {"LAI", "Cab", "Cw", "Cdm", "Vcmax25", "BallBerrySlope"}:
        return max(median_value, 1e-6)
    return median_value


def _apply_spatial_mask(output: xr.Dataset, spatial_valid: xr.DataArray | None) -> xr.Dataset:
    """Restore the authored field footprint on simulation outputs."""
    if spatial_valid is None:
        return output

    masked = output.copy()
    for name, data_array in masked.data_vars.items():
        if {"y", "x"}.issubset(data_array.dims) and _is_numeric_dataarray(data_array):
            masked[name] = data_array.where(spatial_valid)
    return masked


def _validate_hidden_energy_balance_inputs(scope_dataset: xr.Dataset) -> None:
    """Check energy-balance inputs that upstream validation does not enforce."""
    missing = [name for name in ENERGY_BALANCE_REQUIRED_VARS if name not in scope_dataset]
    missing.extend(
        name
        for name in ENERGY_BALANCE_FLUORESCENCE_VARS
        if name not in scope_dataset
    )
    if missing:
        raise ValueError(
            "Coupled energy-balance workflow is missing required variables: "
            + ", ".join(sorted(set(missing)))
        )


def _augment_scope_dataset(dataset: xr.Dataset, config: PipelineConfig) -> xr.Dataset:
    """Add repo-owned forcing variables needed by richer SCOPE workflows.

    ``scope-rtm`` ships a generic dataset preparation function that covers the
    reflectance-era core variables. The fluorescence and thermal branches need
    additional forcing that ARC-SCOPE can derive from the prepared dataset:

    - spectral direct/diffuse irradiance from broadband ``Rin`` and solar angle
    - a diagnostic fluorescence efficiency field ``fqe``
    - explicit aerodynamic and biochemistry defaults for coupled energy balance
    - diagnostic canopy/soil temperatures only for the standalone thermal workflow

    The repo now distinguishes between:

    - ``thermal``: prescribed canopy/soil temperatures for thermal radiance only
    - ``energy-balance``: upstream coupled fluorescence + thermal solvers
    """
    scope_options = config.resolved_scope_options
    is_energy_balance = config.scope_workflow == "energy-balance"
    needs_fluorescence = bool(scope_options.get("calc_fluor"))
    needs_thermal = bool(scope_options.get("calc_planck"))
    needs_spectral_forcing = needs_fluorescence or needs_thermal or is_energy_balance
    if not needs_spectral_forcing:
        return dataset

    augmented = dataset.copy()

    if needs_spectral_forcing:
        spectral_forcing = build_scope_spectral_forcing(
            rin=augmented["Rin"],
            sza=augmented["tts"],
            time_coord=augmented.coords["time"],
            atmos_file=augmented.attrs.get("atmos_file"),
            scope_root_path=config.scope_root_path,
        )
        for name, data_array in spectral_forcing.data_vars.items():
            augmented[name] = data_array.astype(np.float64, copy=False)

    if is_energy_balance:
        energy_balance_state = _energy_balance_state(augmented, config)
        for name, data_array in energy_balance_state.data_vars.items():
            if name not in augmented:
                augmented[name] = data_array.astype(np.float64, copy=False)
        if "fqe" not in augmented:
            augmented["fqe"] = _diagnostic_fqe(augmented)
        return augmented

    if needs_fluorescence and "fqe" not in augmented:
        augmented["fqe"] = _diagnostic_fqe(augmented)

    if needs_thermal:
        thermal_state = _diagnostic_thermal_state(augmented)
        for name, data_array in thermal_state.data_vars.items():
            augmented[name] = data_array.astype(np.float64, copy=False)

    return augmented


def _energy_balance_state(dataset: xr.Dataset, config: PipelineConfig) -> xr.Dataset:
    """Derive or default the extra state needed by the coupled solver.

    ``scope-rtm`` does not currently prepare these variables from the generic
    input dataset. ARC-SCOPE provides transparent defaults anchored to SCOPE's
    own example values plus simple canopy-geometry heuristics for crop height,
    roughness, and soil resistance.
    """
    lai = dataset["LAI"].clip(min=0.0)
    smc = dataset["SMC"] if "SMC" in dataset else _constant_like(lai, 0.25)
    crop_height_cap = CROP_HEIGHT_CAP_M.get(config.crop_type.lower(), 1.5)

    h = (0.12 + 0.18 * lai).clip(min=0.1, max=crop_height_cap)
    d = (0.67 * h).clip(min=0.05)
    z0m = (0.13 * h).clip(min=0.01)
    z = xr.where(h + 2.0 > 10.0, h + 2.0, 10.0)
    rss = (500.0 - 350.0 * _normalise_state(smc)).clip(min=75.0, max=500.0)

    return xr.Dataset(
        {
            "Ca": _constant_like(lai, ENERGY_BALANCE_DEFAULTS["Ca"]),
            "Oa": _constant_like(lai, ENERGY_BALANCE_DEFAULTS["Oa"]),
            "z": z,
            "Cd": _constant_like(lai, ENERGY_BALANCE_DEFAULTS["Cd"]),
            "rwc": _constant_like(lai, ENERGY_BALANCE_DEFAULTS["rwc"]),
            "z0m": z0m,
            "d": d,
            "h": h,
            "rss": rss,
            "rbs": _constant_like(lai, ENERGY_BALANCE_DEFAULTS["rbs"]),
            "Vcmax25": _constant_like(lai, ENERGY_BALANCE_DEFAULTS["Vcmax25"]),
            "BallBerrySlope": _constant_like(
                lai,
                ENERGY_BALANCE_DEFAULTS["BallBerrySlope"],
            ),
        }
    )


def _diagnostic_fqe(dataset: xr.Dataset) -> xr.DataArray:
    """Build a bounded fluorescence-efficiency field from retrieval state."""
    cab_norm = _normalise_state(dataset["Cab"])
    cw_norm = _normalise_state(dataset["Cw"])
    lai_norm = _normalise_state(dataset["LAI"])
    fqe = 0.006 + 0.012 * (0.5 * cab_norm + 0.35 * cw_norm + 0.15 * lai_norm)
    return fqe.clip(min=0.005, max=0.02).rename("fqe")


def _diagnostic_thermal_state(dataset: xr.Dataset) -> xr.Dataset:
    """Approximate canopy and soil temperatures for SCOPE thermal radiance.

    The thermal workflow in ``scope-rtm`` expects explicit sunlit/shaded canopy
    and soil temperatures. ARC-SCOPE does not yet solve the full energy balance
    in-repo, so the thermal demonstration uses a transparent diagnostic forcing
    derived from air temperature, shortwave load, canopy density, wind, leaf
    water, and soil moisture. The resulting temperatures remain spatially and
    temporally varying across the field.
    """
    ta = dataset["Ta"]
    rin = dataset["Rin"].clip(min=0.0)
    wind = dataset["u"] if "u" in dataset else xr.zeros_like(ta) + 2.0
    lai = dataset["LAI"].clip(min=0.0)
    cw = dataset["Cw"].clip(min=0.0)
    smc = dataset["SMC"] if "SMC" in dataset else xr.zeros_like(lai) + 0.25

    solar_load = (rin / 900.0).clip(min=0.0, max=1.5)
    wind_cooling = 1.0 / (1.0 + 0.18 * wind.clip(min=0.0))
    canopy_cover = 1.0 - np.exp(-0.55 * lai)
    soil_exposure = np.exp(-0.65 * lai)
    water_cooling = 1.1 - 0.2 * _normalise_state(cw)
    soil_dryness = 1.15 - 0.3 * _normalise_state(smc)

    tcu = ta + 0.4 + 4.2 * solar_load * wind_cooling * water_cooling * (0.35 + 0.65 * canopy_cover)
    tch = ta + 0.2 + 2.1 * solar_load * wind_cooling * water_cooling * (0.25 + 0.75 * canopy_cover)
    tsu = ta + 1.0 + 7.0 * solar_load * soil_exposure * soil_dryness / (1.0 + 0.12 * wind.clip(min=0.0))
    tsh = ta + 0.5 + 3.4 * solar_load * soil_exposure * soil_dryness / (1.0 + 0.18 * wind.clip(min=0.0))

    return xr.Dataset(
        {
            "Tcu": tcu.clip(min=-20.0, max=60.0),
            "Tch": tch.clip(min=-20.0, max=60.0),
            "Tsu": tsu.clip(min=-20.0, max=75.0),
            "Tsh": tsh.clip(min=-20.0, max=75.0),
        }
    )


def _normalise_state(data_array: xr.DataArray) -> xr.DataArray:
    """Normalise a field to 0-1 while remaining robust to constant inputs."""
    finite = data_array.where(np.isfinite(data_array))
    min_value = float(finite.min(skipna=True).item())
    max_value = float(finite.max(skipna=True).item())
    if not np.isfinite(min_value) or not np.isfinite(max_value) or abs(max_value - min_value) < 1e-9:
        return xr.zeros_like(data_array, dtype=np.float64) + 0.5
    return ((data_array - min_value) / (max_value - min_value)).clip(min=0.0, max=1.0)


def _constant_like(data_array: xr.DataArray, value: float) -> xr.DataArray:
    """Return a floating-point field with the same shape and coordinates."""
    return xr.zeros_like(data_array, dtype=np.float64) + value


def _is_numeric_dataarray(data_array: xr.DataArray) -> bool:
    """Return whether a DataArray contains numeric values."""
    return np.issubdtype(data_array.dtype, np.number)


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
