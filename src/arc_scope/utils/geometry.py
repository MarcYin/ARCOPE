"""Solar and viewing geometry computations.

Computes solar zenith/azimuth angles from date, time, and geographic location.
These are needed to build the observation dataset that SCOPE requires.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np


def solar_position(
    lat: float | np.ndarray,
    lon: float | np.ndarray,
    dt: datetime | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute solar zenith and azimuth angles.

    Uses the simplified solar position algorithm suitable for remote sensing
    applications (Spencer, 1971; Iqbal, 1983).

    Parameters
    ----------
    lat:
        Latitude(s) in degrees (positive north).
    lon:
        Longitude(s) in degrees (positive east).
    dt:
        Datetime(s) for the calculation.

    Returns
    -------
    sza:
        Solar zenith angle(s) in degrees.
    saa:
        Solar azimuth angle(s) in degrees (clockwise from north).
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    if isinstance(dt, datetime):
        doy = dt.timetuple().tm_yday
        hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    else:
        dt = np.asarray(dt, dtype="datetime64[s]")
        doy = (dt - dt.astype("datetime64[Y]")).astype("timedelta64[D]").astype(int) + 1
        hour = (dt - dt.astype("datetime64[D]")).astype("timedelta64[s]").astype(float) / 3600.0

    doy = np.asarray(doy, dtype=np.float64)
    hour = np.asarray(hour, dtype=np.float64)

    # Solar declination (Spencer, 1971)
    gamma = 2 * np.pi * (doy - 1) / 365.0
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )

    # Equation of time (minutes)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.04089 * np.sin(2 * gamma)
    )

    # Solar hour angle
    solar_time = hour + (eqtime + 4.0 * lon) / 60.0
    ha = np.radians((solar_time - 12.0) * 15.0)

    lat_rad = np.radians(lat)

    # Solar zenith angle
    cos_sza = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(ha)
    cos_sza = np.clip(cos_sza, -1.0, 1.0)
    sza = np.degrees(np.arccos(cos_sza))

    # Solar azimuth angle
    sin_sza = np.sin(np.radians(sza))
    sin_sza = np.where(sin_sza == 0, 1e-10, sin_sza)
    cos_saa = (np.sin(decl) - np.cos(np.radians(sza)) * np.sin(lat_rad)) / (sin_sza * np.cos(lat_rad))
    cos_saa = np.clip(cos_saa, -1.0, 1.0)
    saa = np.degrees(np.arccos(cos_saa))
    saa = np.where(ha > 0, 360.0 - saa, saa)

    return sza, saa


def relative_azimuth(
    solar_azimuth: float | np.ndarray,
    viewing_azimuth: float | np.ndarray,
) -> np.ndarray:
    """Compute relative azimuth angle between sun and sensor.

    Parameters
    ----------
    solar_azimuth:
        Solar azimuth angle(s) in degrees.
    viewing_azimuth:
        Sensor viewing azimuth angle(s) in degrees.

    Returns
    -------
    Relative azimuth angle(s) in degrees [0, 360).
    """
    return (np.asarray(viewing_azimuth) - np.asarray(solar_azimuth)) % 360.0
