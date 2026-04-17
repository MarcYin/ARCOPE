"""Shortwave radiation partitioning into direct and diffuse components.

SCOPE's fluorescence and energy-balance workflows require spectrally-resolved
direct (``Esun_sw``) and diffuse (``Esky_sw``) irradiance on its wavelength
grid.  This module partitions total incoming shortwave (``Rin``) into these
components using the BRL (Boland-Ridley-Lauret) diffuse fraction model and
rescales the bundled SCOPE reference spectra onto the requested wavelength
grids.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr


def diffuse_fraction_brl(
    kt: np.ndarray,
    *,
    a: float = -5.38,
    b: float = 6.63,
    c: float = 0.006,
    d: float = -0.007,
) -> np.ndarray:
    """Estimate diffuse fraction using the BRL logistic model.

    Parameters
    ----------
    kt:
        Clearness index (Rin / extraterrestrial irradiance), clipped to [0, 1].

    Returns
    -------
    Diffuse fraction ``kd = Rdiffuse / Rtotal``, in [0, 1].
    """
    kt = np.clip(np.asarray(kt, dtype=np.float64), 0.0, 1.0)
    kd = 1.0 / (1.0 + np.exp(a + b * kt))
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

    kd = diffuse_fraction_brl(kt)
    diffuse = rin * kd
    direct = rin - diffuse

    return np.maximum(direct, 0.0), np.maximum(diffuse, 0.0)


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


def build_scope_spectral_forcing(
    rin: xr.DataArray,
    sza: xr.DataArray,
    *,
    time_coord: xr.DataArray,
    atmos_file: str | Path | None = None,
    scope_root_path: str | Path | None = None,
    wavelength_nm: np.ndarray | None = None,
    excitation_wavelength_nm: np.ndarray | None = None,
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
    direct, diffuse = partition_shortwave(
        rin_aligned.values,
        sza_aligned.values,
        doy_aligned.values,
    )

    radiation_dir = resolve_scope_radiation_dir(
        atmos_file=atmos_file,
        scope_root_path=scope_root_path,
    )
    ref_wl, ref_direct, ref_diffuse = _load_scope_reference_spectra(str(radiation_dir))
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
    excitation_direct = normalised_reference_spectrum(
        excitation_wavelength_nm,
        reference_wavelength_nm=ref_wl,
        reference_flux=ref_direct,
    )
    excitation_diffuse = normalised_reference_spectrum(
        excitation_wavelength_nm,
        reference_wavelength_nm=ref_wl,
        reference_flux=ref_diffuse,
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

    return xr.Dataset(
        {
            "Esun_sw": direct_da * wavelength_da,
            "Esky_sw": diffuse_da * wavelength_diffuse_da,
            "Esun_": direct_da * excitation_da,
            "Esky_": diffuse_da * excitation_diffuse_da,
        }
    )
