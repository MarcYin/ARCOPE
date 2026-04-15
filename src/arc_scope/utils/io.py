"""Load/save helpers for common file formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from arc_scope.utils.types import PathLike


def load_geojson_bounds(geojson_path: PathLike) -> tuple[float, float, float, float]:
    """Extract bounding box (minx, miny, maxx, maxy) from a GeoJSON file.

    Parameters
    ----------
    geojson_path:
        Path to a GeoJSON file containing one or more features.

    Returns
    -------
    Tuple of (minx, miny, maxx, maxy) in the GeoJSON's CRS (typically WGS84).
    """
    with open(geojson_path) as f:
        data = json.load(f)

    coords = _extract_all_coordinates(data)
    if not coords:
        raise ValueError(f"No coordinates found in {geojson_path}")

    arr = np.array(coords)
    return float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 0].max()), float(arr[:, 1].max())


def load_geojson_centroid(geojson_path: PathLike) -> tuple[float, float]:
    """Extract centroid (lon, lat) from a GeoJSON file."""
    minx, miny, maxx, maxy = load_geojson_bounds(geojson_path)
    return (minx + maxx) / 2.0, (miny + maxy) / 2.0


def _extract_all_coordinates(geojson: dict[str, Any]) -> list[list[float]]:
    """Recursively extract all coordinate pairs from a GeoJSON object."""
    coords: list[list[float]] = []

    if "coordinates" in geojson:
        _flatten_coords(geojson["coordinates"], coords)
    if "geometries" in geojson:
        for geom in geojson["geometries"]:
            coords.extend(_extract_all_coordinates(geom))
    if "features" in geojson:
        for feature in geojson["features"]:
            if "geometry" in feature and feature["geometry"]:
                coords.extend(_extract_all_coordinates(feature["geometry"]))
    if "geometry" in geojson and isinstance(geojson["geometry"], dict):
        coords.extend(_extract_all_coordinates(geojson["geometry"]))

    return coords


def _flatten_coords(obj: Any, out: list[list[float]]) -> None:
    """Recursively flatten nested coordinate arrays."""
    if not obj:
        return
    if isinstance(obj[0], (int, float)):
        out.append(list(obj[:2]))
    else:
        for item in obj:
            _flatten_coords(item, out)
