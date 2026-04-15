"""Shortwave radiation partitioning into direct and diffuse components.

SCOPE's fluorescence and energy-balance workflows require spectrally-resolved
direct (``Esun_sw``) and diffuse (``Esky_sw``) irradiance on its wavelength
grid.  This module partitions total incoming shortwave (``Rin``) into these
components using the BRL (Boland-Ridley-Lauret) diffuse fraction model.
"""

from __future__ import annotations

import numpy as np


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
