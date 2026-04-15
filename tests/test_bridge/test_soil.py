"""Tests for soil parameter validation."""

import numpy as np
import pytest

from arc_scope.bridge.soil import validate_soil_params


def test_valid_params_pass():
    """In-range parameters should pass without modification."""
    result = validate_soil_params(
        brightness=np.array([0.3, 0.5]),
        lat=np.array([20.0, 25.0]),
        lon=np.array([30.0, 50.0]),
        smc=np.array([20.0, 60.0]),
    )
    assert set(result.keys()) == {"BSMBrightness", "BSMlat", "BSMlon", "SMC"}
    np.testing.assert_array_equal(result["BSMBrightness"], [0.3, 0.5])


def test_clip_out_of_range():
    """Out-of-range values should be clipped by default."""
    result = validate_soil_params(
        brightness=np.array([0.0, 1.0]),  # Range [0.1, 0.7]
        lat=np.array([5.0, 35.0]),         # Range [10, 30]
        lon=np.array([5.0, 80.0]),         # Range [10, 70]
        smc=np.array([0.0, 110.0]),        # Range [2, 100]
    )
    assert result["BSMBrightness"].min() >= 0.1
    assert result["BSMBrightness"].max() <= 0.7
    assert result["SMC"].max() <= 100.0


def test_strict_raises():
    """Strict mode should raise on out-of-range values."""
    with pytest.raises(ValueError, match="BSMBrightness"):
        validate_soil_params(
            brightness=np.array([0.0]),
            lat=np.array([20.0]),
            lon=np.array([30.0]),
            smc=np.array([50.0]),
            strict=True,
        )
