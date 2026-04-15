"""Tests for solar/viewing geometry computations."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from arc_scope.utils.geometry import relative_azimuth, solar_position


# ---------------------------------------------------------------------------
# solar_position tests
# ---------------------------------------------------------------------------


def test_solar_position_noon_equator():
    """At equinox noon on the equator the sun should be nearly overhead (sza ~ 0)."""
    # March 20 equinox, 12:00 UTC at lon=0 (solar noon)
    dt = datetime(2021, 3, 20, 12, 0, 0)
    sza, saa = solar_position(0.0, 0.0, dt)
    # SZA should be close to 0; allow a few degrees for equation-of-time offset
    assert float(sza) < 5.0, f"Expected SZA near 0 at equinox noon, got {float(sza)}"


def test_solar_position_polar_winter():
    """During polar night the solar zenith angle must exceed 90 degrees."""
    # December 21 at 70N, midnight UTC
    dt = datetime(2021, 12, 21, 0, 0, 0)
    sza, _saa = solar_position(70.0, 25.0, dt)
    assert float(sza) > 90.0, f"Expected SZA > 90 during polar night, got {float(sza)}"


def test_solar_position_array_input():
    """solar_position should accept array-valued lat/lon and vectorised datetime64."""
    lats = np.array([0.0, 45.0, -30.0])
    lons = np.array([0.0, 10.0, -50.0])
    dts = np.array(
        ["2021-06-21T12:00:00", "2021-06-21T12:00:00", "2021-06-21T12:00:00"],
        dtype="datetime64[s]",
    )
    sza, saa = solar_position(lats, lons, dts)
    assert sza.shape == (3,)
    assert saa.shape == (3,)


def test_solar_position_returns_float_arrays():
    """Returned angles must be numpy floating-point types, even for scalar inputs."""
    dt = datetime(2021, 6, 21, 12, 0, 0)
    sza, saa = solar_position(52.0, 5.0, dt)
    # For scalar inputs the function returns a 0-d numpy float (np.float64),
    # which is a numpy generic but not necessarily ndim >= 1.
    assert isinstance(sza, (np.ndarray, np.floating))
    assert isinstance(saa, (np.ndarray, np.floating))
    assert np.asarray(sza).dtype.kind == "f"
    assert np.asarray(saa).dtype.kind == "f"


# ---------------------------------------------------------------------------
# relative_azimuth tests
# ---------------------------------------------------------------------------


def test_relative_azimuth_same_direction():
    """When sun and sensor point the same way the relative azimuth should be ~0."""
    result = relative_azimuth(180.0, 180.0)
    assert float(result) == pytest.approx(0.0, abs=1e-10)


def test_relative_azimuth_opposite():
    """Opposite directions should yield 180 degrees."""
    result = relative_azimuth(0.0, 180.0)
    assert float(result) == pytest.approx(180.0, abs=1e-10)


def test_relative_azimuth_wraps_360():
    """Result should always be in [0, 360) via modulo wrapping."""
    result = relative_azimuth(350.0, 10.0)
    assert 0.0 <= float(result) < 360.0
    # 10 - 350 = -340 -> -340 % 360 = 20
    assert float(result) == pytest.approx(20.0, abs=1e-10)


def test_relative_azimuth_negative_input():
    """Negative input azimuths should still produce a valid [0, 360) result."""
    result = relative_azimuth(-90.0, 90.0)
    assert 0.0 <= float(result) < 360.0
    # 90 - (-90) = 180
    assert float(result) == pytest.approx(180.0, abs=1e-10)
