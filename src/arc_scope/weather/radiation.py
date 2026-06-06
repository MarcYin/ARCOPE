"""Shortwave radiation partitioning into direct and diffuse components.

SCOPE's fluorescence and energy-balance workflows require spectrally-resolved
direct (``Esun_sw``) and diffuse (``Esky_sw``) irradiance on its wavelength
grid.  This module partitions total incoming shortwave (``Rin``) into these
components and rescales the bundled SCOPE reference spectra onto the requested
wavelength grids.  Two direct/diffuse split models are available:

* ``"erbs"`` (default) -- the Erbs (1982) clearness-index model, which varies the
  diffuse fraction per timestep with cloudiness (more physical for real weather).
* ``"modtran"`` -- the fixed MODTRAN ``.atm`` split recovered from the bundled
  reference spectra (~0.275 for FLEX-S3), matching SCOPE's ``calcTOCirr``; use for
  MATLAB-grid consistency.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr


def diffuse_fraction_erbs(kt: np.ndarray) -> np.ndarray:
    """Estimate diffuse fraction from the clearness index (Erbs et al., 1982).

    A single-predictor (``kt``-only) correlation calibrated for hourly data:

        kt <= 0.22      kd = 1.0 - 0.09*kt
        0.22 < kt<=0.80 kd = 0.9511 - 0.1604*kt + 4.388*kt^2
                              - 16.638*kt^3 + 12.336*kt^4
        kt > 0.80       kd = 0.165

    This replaces the previous use of the Boland-Ridley-Lauret (BRL) *logistic*
    coefficients in a ``kt``-only form. Those coefficients were fit jointly with
    four other predictors (solar altitude, apparent solar time, daily clearness,
    persistence); applying them to ``kt`` alone is not valid and over-predicts the
    diffuse fraction badly (~0.67 vs ~0.27 on clear days). Erbs is properly
    calibrated for ``kt`` alone and agrees with the full BRL and with SCOPE's own
    standard-atmosphere split (~0.27-0.28 diffuse on clear days).

    Parameters
    ----------
    kt:
        Clearness index (Rin / horizontal extraterrestrial irradiance), [0, 1].

    Returns
    -------
    Diffuse fraction ``kd = Rdiffuse / Rtotal``, in [0, 1].
    """
    kt = np.clip(np.asarray(kt, dtype=np.float64), 0.0, 1.0)
    poly = 0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4
    kd = np.where(kt <= 0.22, 1.0 - 0.09 * kt, np.where(kt <= 0.80, poly, 0.165))
    return np.clip(kd, 0.0, 1.0)


def extraterrestrial_irradiance(doy: np.ndarray) -> np.ndarray:
    """Top-of-atmosphere solar irradiance accounting for Earth-Sun distance.

    Parameters
    ----------
    doy:
        Day of year (1-366).

    Returns
    -------
    Extraterrestrial irradiance (W m-2).
    """
    doy = np.asarray(doy, dtype=np.float64)
    solar_constant = 1361.0  # W m-2
    # Spencer (1971) correction for Earth-Sun distance
    gamma = 2 * np.pi * (doy - 1) / 365.0
    correction = (
        1.00011
        + 0.034221 * np.cos(gamma)
        + 0.001280 * np.sin(gamma)
        + 0.000719 * np.cos(2 * gamma)
        + 0.000077 * np.sin(2 * gamma)
    )
    return solar_constant * correction


def partition_shortwave(
    rin: np.ndarray,
    sza: np.ndarray,
    doy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition total shortwave into direct and diffuse components.

    Parameters
    ----------
    rin:
        Total incoming shortwave radiation (W m-2).
    sza:
        Solar zenith angle (degrees).
    doy:
        Day of year.

    Returns
    -------
    direct:
        Direct (beam) shortwave (W m-2).
    diffuse:
        Diffuse shortwave (W m-2).
    """
    rin = np.asarray(rin, dtype=np.float64)
    sza = np.asarray(sza, dtype=np.float64)

    cos_sza = np.cos(np.radians(sza))
    cos_sza = np.clip(cos_sza, 0.01, 1.0)  # Avoid division by zero

    # Horizontal extraterrestrial irradiance
    i0 = extraterrestrial_irradiance(doy) * cos_sza

    # Clearness index
    kt = np.where(i0 > 0, rin / i0, 0.0)
    kt = np.clip(kt, 0.0, 1.0)

    kd = diffuse_fraction_erbs(kt)
    diffuse = rin * kd
    direct = rin - diffuse

    return np.maximum(direct, 0.0), np.maximum(diffuse, 0.0)


def diffuse_fraction_from_reference(
    reference_wavelength_nm: np.ndarray,
    reference_direct: np.ndarray,
    reference_diffuse: np.ndarray,
    *,
    optical_max_nm: float = 3000.0,
) -> float:
    """Native diffuse fraction of the bundled MODTRAN reference spectra.

    SCOPE's ``calcTOCirr`` fixes the direct/diffuse split from the MODTRAN ``.atm``
    file (the ``Esun_``/``Esky_`` shapes) and only rescales the *magnitude* to
    ``Rin``; it does not re-estimate the split per timestep. This recovers that
    fixed split by integrating the bundled ``Esun_``/``Esky_`` spectra over the
    optical band (``wavelength < 3000 nm``). For the standard FLEX-S3 atmosphere
    it is ~0.275 — the same value MATLAB SCOPE uses, independent of clearness.
    """
    wl = np.asarray(reference_wavelength_nm, dtype=np.float64)
    opt = wl < optical_max_nm
    direct_int = float(np.trapezoid(np.clip(np.asarray(reference_direct, dtype=np.float64), 0.0, None)[opt], wl[opt]))
    diffuse_int = float(np.trapezoid(np.clip(np.asarray(reference_diffuse, dtype=np.float64), 0.0, None)[opt], wl[opt]))
    total = direct_int + diffuse_int
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Reference spectra must integrate to a positive total.")
    return diffuse_int / total


@lru_cache(maxsize=8)
def _load_scope_reference_spectra(reference_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the direct and diffuse SCOPE reference spectra from disk.

    The upstream SCOPE assets ship ``Esun_.dat`` and ``Esky_.dat`` as plain
    vectors without an explicit wavelength column. The legacy files map to a
    1 nm grid starting at 400 nm. We resample those shapes onto the modern
    ``scope-rtm`` wavelength grids and scale them to the requested broadband
    fluxes.
    """
    reference_path = Path(reference_dir)
    direct = np.loadtxt(reference_path / "Esun_.dat", dtype=np.float64)
    diffuse = np.loadtxt(reference_path / "Esky_.dat", dtype=np.float64)
    wavelength = 400.0 + np.arange(direct.size, dtype=np.float64)
    return wavelength, direct, diffuse


def resolve_scope_radiation_dir(*, atmos_file: str | Path | None, scope_root_path: str | Path | None) -> Path:
    """Resolve the upstream SCOPE ``radiationdata`` directory."""
    if atmos_file is not None:
        return Path(atmos_file).expanduser().resolve().parent

    if scope_root_path is not None:
        root = Path(scope_root_path).expanduser()
        candidate = root / "input" / "radiationdata"
        if candidate.exists():
            return candidate.resolve()
        fallback = root / "radiationdata"
        if fallback.exists():
            return fallback.resolve()

    raise ValueError(
        "Unable to resolve SCOPE radiationdata directory. "
        "Provide either a dataset attrs['atmos_file'] value or scope_root_path."
    )


def normalised_reference_spectrum(
    target_wavelength_nm: np.ndarray,
    *,
    reference_wavelength_nm: np.ndarray,
    reference_flux: np.ndarray,
) -> np.ndarray:
    """Interpolate and normalise a reference spectrum to unit broadband flux."""
    target_wavelength_nm = np.asarray(target_wavelength_nm, dtype=np.float64)
    reference_wavelength_nm = np.asarray(reference_wavelength_nm, dtype=np.float64)
    reference_flux = np.asarray(reference_flux, dtype=np.float64)

    interpolated = np.interp(
        target_wavelength_nm,
        reference_wavelength_nm,
        np.clip(reference_flux, a_min=0.0, a_max=None),
        left=0.0,
        right=0.0,
    )
    wavelength_um = target_wavelength_nm / 1000.0
    total = np.trapezoid(interpolated, x=wavelength_um)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Reference irradiance spectrum must integrate to a positive value.")
    return interpolated / total


def _full_band_normalised_spectrum(
    target_wavelength_nm: np.ndarray,
    *,
    full_wavelength_nm: np.ndarray,
    reference_wavelength_nm: np.ndarray,
    reference_flux: np.ndarray,
) -> np.ndarray:
    """Sample a reference spectrum, normalised by its integral over the FULL band.

    Unlike :func:`normalised_reference_spectrum` (which normalises over the target
    grid), this divides by the integral over ``full_wavelength_nm`` so that a
    sub-band (e.g. the 400-750 nm excitation band) carries only its real share of
    the broadband flux when scaled by a broadband total.
    """
    ref_wl = np.asarray(reference_wavelength_nm, dtype=np.float64)
    ref_flux = np.clip(np.asarray(reference_flux, dtype=np.float64), a_min=0.0, a_max=None)
    full_wl = np.asarray(full_wavelength_nm, dtype=np.float64)
    total = np.trapezoid(np.interp(full_wl, ref_wl, ref_flux, left=0.0, right=0.0), x=full_wl / 1000.0)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Reference irradiance spectrum must integrate to a positive value.")
    sampled = np.interp(np.asarray(target_wavelength_nm, dtype=np.float64), ref_wl, ref_flux, left=0.0, right=0.0)
    return sampled / total


def normalised_planck_shape(wavelength_nm: np.ndarray, temperature_k: float = 288.0) -> np.ndarray:
    """Normalised blackbody spectral shape that integrates to 1 over wavelength (in um)."""
    wl_nm = np.asarray(wavelength_nm, dtype=np.float64)
    c2 = 1.438777e7  # second radiation constant hc/k, in nm*K
    shape = 1.0 / (wl_nm**5 * (np.exp(c2 / (wl_nm * temperature_k)) - 1.0))
    total = np.trapezoid(shape, x=wl_nm / 1000.0)
    return shape / total


def build_scope_spectral_forcing(
    rin: xr.DataArray,
    sza: xr.DataArray,
    *,
    time_coord: xr.DataArray,
    atmos_file: str | Path | None = None,
    scope_root_path: str | Path | None = None,
    wavelength_nm: np.ndarray | None = None,
    excitation_wavelength_nm: np.ndarray | None = None,
    rli: xr.DataArray | None = None,
    thermal_wavelength_nm: np.ndarray | None = None,
    sky_temperature_k: float = 288.0,
    diffuse_model: str = "erbs",
) -> xr.Dataset:
    """Build SCOPE spectral irradiance inputs from broadband shortwave forcing.

    Parameters
    ----------
    rin:
        Incoming shortwave radiation with a ``time`` dimension.
    sza:
        Solar zenith angle on the same time grid as ``rin``.
    time_coord:
        Dataset time coordinate used to derive day-of-year values.
    atmos_file, scope_root_path:
        SCOPE asset references used to locate the bundled irradiance shapes.
    wavelength_nm:
        Target shortwave wavelength grid. Defaults to 400-2400 nm at 1 nm.
    excitation_wavelength_nm:
        Target fluorescence excitation grid. Defaults to 400-750 nm at 5 nm.

    Returns
    -------
    xr.Dataset
        Dataset containing ``Esun_sw``, ``Esky_sw``, ``Esun_``, and ``Esky_``.
    """
    wavelength_nm = (
        np.asarray(wavelength_nm, dtype=np.float64)
        if wavelength_nm is not None
        else np.arange(400.0, 2401.0, 1.0, dtype=np.float64)
    )
    excitation_wavelength_nm = (
        np.asarray(excitation_wavelength_nm, dtype=np.float64)
        if excitation_wavelength_nm is not None
        else np.arange(400.0, 751.0, 5.0, dtype=np.float64)
    )

    time_index = time_coord.to_index()
    doy = xr.DataArray(
        time_index.dayofyear.astype(np.float64),
        dims=("time",),
        coords={"time": time_coord.values},
    )
    rin_aligned, sza_aligned, doy_aligned = xr.broadcast(rin, sza, doy)

    radiation_dir = resolve_scope_radiation_dir(
        atmos_file=atmos_file,
        scope_root_path=scope_root_path,
    )
    ref_wl, ref_direct, ref_diffuse = _load_scope_reference_spectra(str(radiation_dir))

    if diffuse_model == "modtran":
        # Match SCOPE's calcTOCirr: the direct/diffuse split is fixed by the
        # MODTRAN .atm (recovered from the bundled spectra), only the magnitude is
        # rescaled to Rin. Clear-sky split (~0.275), independent of clearness.
        kd = diffuse_fraction_from_reference(ref_wl, ref_direct, ref_diffuse)
        rin_vals = np.maximum(rin_aligned.values, 0.0)
        diffuse = rin_vals * kd
        direct = rin_vals * (1.0 - kd)
    elif diffuse_model == "erbs":
        # Per-timestep clearness-index split (Erbs 1982): more physical for real
        # cloudy-sky weather forcing, where the diffuse fraction varies.
        direct, diffuse = partition_shortwave(
            rin_aligned.values,
            sza_aligned.values,
            doy_aligned.values,
        )
    else:
        raise ValueError(f"unknown diffuse_model {diffuse_model!r}; expected 'erbs' or 'modtran'")
    shortwave_direct = normalised_reference_spectrum(
        wavelength_nm,
        reference_wavelength_nm=ref_wl,
        reference_flux=ref_direct,
    )
    shortwave_diffuse = normalised_reference_spectrum(
        wavelength_nm,
        reference_wavelength_nm=ref_wl,
        reference_flux=ref_diffuse,
    )
    # Fluorescence excitation irradiance. This must carry only the share of the
    # broadband shortwave that actually falls in the 400-750 nm excitation band
    # (~45% of broadband), NOT the whole broadband. Normalising the spectrum over
    # the excitation band alone (as a stand-alone call to
    # ``normalised_reference_spectrum`` would) forces 100% of ``direct`` into
    # 400-750 nm and roughly doubles the excitation irradiance, doubling SIF. We
    # instead normalise over the full shortwave band and sample at the excitation
    # wavelengths, so ``Esun_``/``Esky_`` carry the real PAR-band irradiance.
    excitation_direct = _full_band_normalised_spectrum(
        excitation_wavelength_nm, full_wavelength_nm=wavelength_nm,
        reference_wavelength_nm=ref_wl, reference_flux=ref_direct,
    )
    excitation_diffuse = _full_band_normalised_spectrum(
        excitation_wavelength_nm, full_wavelength_nm=wavelength_nm,
        reference_wavelength_nm=ref_wl, reference_flux=ref_diffuse,
    )

    direct_da = xr.DataArray(direct, dims=rin_aligned.dims, coords=rin_aligned.coords)
    diffuse_da = xr.DataArray(diffuse, dims=rin_aligned.dims, coords=rin_aligned.coords)
    wavelength_da = xr.DataArray(shortwave_direct, dims=("wavelength",), coords={"wavelength": wavelength_nm})
    wavelength_diffuse_da = xr.DataArray(shortwave_diffuse, dims=("wavelength",), coords={"wavelength": wavelength_nm})
    excitation_da = xr.DataArray(
        excitation_direct,
        dims=("excitation_wavelength",),
        coords={"excitation_wavelength": excitation_wavelength_nm},
    )
    excitation_diffuse_da = xr.DataArray(
        excitation_diffuse,
        dims=("excitation_wavelength",),
        coords={"excitation_wavelength": excitation_wavelength_nm},
    )

    out = xr.Dataset(
        {
            "Esun_sw": direct_da * wavelength_da,
            "Esky_sw": diffuse_da * wavelength_diffuse_da,
            "Esun_": direct_da * excitation_da,
            "Esky_": diffuse_da * excitation_diffuse_da,
        }
    )

    # Incoming atmospheric longwave. SCOPE's energy balance zero-fills the thermal
    # band when ``Esun_lw``/``Esky_lw`` are absent, which drops the incoming
    # longwave entirely and runs the canopy ~10-15 deg C too cold. Atmospheric
    # thermal emission is all-sky (diffuse), so put the broadband ``Rli`` into
    # ``Esky_lw`` distributed over the thermal band by a normalised blackbody, and
    # leave ``Esun_lw`` at zero. ``Esky_lw`` integrates (over wavelength in um) to
    # ``Rli``, matching the shortwave convention.
    if rli is not None:
        thermal_wavelength_nm = (
            np.asarray(thermal_wavelength_nm, dtype=np.float64)
            if thermal_wavelength_nm is not None
            else np.concatenate(
                [np.arange(2500.0, 15001.0, 100.0), np.arange(16000.0, 50001.0, 1000.0)]
            )
        )
        planck = normalised_planck_shape(thermal_wavelength_nm, sky_temperature_k)
        rli_aligned, _ = xr.broadcast(rli, doy)
        rli_da = xr.DataArray(rli_aligned.values, dims=rli_aligned.dims, coords=rli_aligned.coords)
        thermal_da = xr.DataArray(
            planck, dims=("thermal_wavelength",), coords={"thermal_wavelength": thermal_wavelength_nm}
        )
        out["Esky_lw"] = rli_da * thermal_da
        out["Esun_lw"] = xr.zeros_like(out["Esky_lw"])

    return out
