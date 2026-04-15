"""Shared type aliases used across arc_scope modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
import xarray as xr

# Path-like types
PathLike = Union[str, Path]

# Bounding box: (minx, miny, maxx, maxy)
BBox = tuple[float, float, float, float]

# Array-like inputs
ArrayLike = Union[np.ndarray, list[float], tuple[float, ...]]

# xarray data types
XrData = Union[xr.Dataset, xr.DataArray]

# CRS value (EPSG int, WKT string, or pyproj CRS object)
CRSLike = Any
