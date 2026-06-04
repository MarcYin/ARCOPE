"""Tests for radiation partitioning."""

import numpy as np
import pytest

from arc_scope.weather.radiation import (
    diffuse_fraction_erbs,
    extraterrestrial_irradiance,
    partition_shortwave,
)


def test_extraterrestrial_irradiance_range():
    """ETR should be ~1361 W/m2 with seasonal variation."""
    doys = np.arange(1, 366)
    etr = extraterrestrial_irradiance(doys)
    assert etr.min() > 1300
    assert etr.max() < 1420


def test_diffuse_fraction_bounds():
    """Diffuse fraction should be in [0, 1]."""
    kt = np.linspace(0, 1, 100)
    kd = diffuse_fraction_erbs(kt)
    assert np.all(kd >= 0)
    assert np.all(kd <= 1)


def test_diffuse_fraction_clear_sky_physical():
    """Erbs must give a physical clear-sky diffuse fraction (~0.2-0.35 at kt~0.7).

    Regression guard for the previous BRL-coefficient miscalibration, which
    returned ~0.67 diffuse on clear days. Erbs (and the full BRL, and SCOPE's
    standard atmosphere) all sit near 0.27.
    """
    assert diffuse_fraction_erbs(np.array([0.70]))[0] == pytest.approx(0.27, abs=0.06)
    # Overcast (low clearness) must be mostly diffuse; very clear must be low.
    assert diffuse_fraction_erbs(np.array([0.15]))[0] > 0.9
    assert diffuse_fraction_erbs(np.array([0.85]))[0] < 0.2
    # Monotonic non-increasing over the mid range.
    kt = np.linspace(0.25, 0.79, 50)
    kd = diffuse_fraction_erbs(kt)
    assert np.all(np.diff(kd) <= 1e-9)


def test_partition_shortwave_conservation():
    """Direct + diffuse should equal total."""
    rin = np.array([500.0, 800.0, 200.0])
    sza = np.array([30.0, 45.0, 60.0])
    doy = np.array([180, 180, 180])

    direct, diffuse = partition_shortwave(rin, sza, doy)
    np.testing.assert_allclose(direct + diffuse, rin, atol=1e-10)


def test_partition_shortwave_non_negative():
    """Both components should be non-negative."""
    rin = np.array([100.0, 0.0])
    sza = np.array([30.0, 85.0])
    doy = np.array([180, 180])

    direct, diffuse = partition_shortwave(rin, sza, doy)
    assert np.all(direct >= 0)
    assert np.all(diffuse >= 0)
