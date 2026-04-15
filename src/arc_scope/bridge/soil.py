"""BSM soil parameter bridge between ARC and SCOPE.

ARC internally normalises soil parameters for its forward model (see
``arc_sample_generator.adjust_soil_params``), but the values stored in
``scale_data`` (columns 11-14) are the *original* physical values before
normalisation.  SCOPE's ``SoilBSMModel`` expects these physical values
directly, so no un-normalisation is needed.

This module provides validation utilities to ensure ARC soil parameters
fall within the ranges expected by SCOPE.
"""

from __future__ import annotations

import numpy as np

from arc_scope.bridge.parameter_map import ARC_SOIL_RANGES


def validate_soil_params(
    brightness: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    smc: np.ndarray,
    *,
    strict: bool = False,
) -> dict[str, np.ndarray]:
    """Validate and optionally clip ARC BSM soil parameters for SCOPE.

    Parameters
    ----------
    brightness:
        Soil brightness (BSMBrightness).
    lat:
        Soil spectral shape parameter 1 (BSMlat).
    lon:
        Soil spectral shape parameter 2 (BSMlon).
    smc:
        Soil volumetric moisture content, % (SMC).
    strict:
        If ``True``, raise ``ValueError`` on out-of-range values.
        If ``False`` (default), clip to valid ranges.

    Returns
    -------
    Dict mapping SCOPE variable names to validated arrays.
    """
    params = {
        "BSMBrightness": (np.asarray(brightness, dtype=np.float64), ARC_SOIL_RANGES["BSMBrightness"]),
        "BSMlat": (np.asarray(lat, dtype=np.float64), ARC_SOIL_RANGES["BSMlat"]),
        "BSMlon": (np.asarray(lon, dtype=np.float64), ARC_SOIL_RANGES["BSMlon"]),
        "SMC": (np.asarray(smc, dtype=np.float64), ARC_SOIL_RANGES["SMC"]),
    }

    result: dict[str, np.ndarray] = {}
    for name, (values, (lo, hi)) in params.items():
        finite = values[np.isfinite(values)]
        if finite.size > 0 and (finite.min() < lo or finite.max() > hi):
            if strict:
                raise ValueError(
                    f"{name} has values outside [{lo}, {hi}]: "
                    f"min={finite.min():.4f}, max={finite.max():.4f}"
                )
            values = np.clip(values, lo, hi)
        result[name] = values

    return result
