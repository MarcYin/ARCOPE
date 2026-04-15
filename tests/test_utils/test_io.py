"""Tests for I/O helpers (GeoJSON loading)."""
from __future__ import annotations

import json
import os

import pytest

from arc_scope.data import TEST_FIELD_GEOJSON
from arc_scope.utils.io import load_geojson_bounds, load_geojson_centroid


# ---------------------------------------------------------------------------
# Tests using the bundled test_field.geojson
# ---------------------------------------------------------------------------


def test_load_geojson_bounds():
    """Bounds of the bundled test field should be near 5.019-5.023 E, 51.275-51.279 N."""
    minx, miny, maxx, maxy = load_geojson_bounds(TEST_FIELD_GEOJSON)
    assert 5.019 < minx < 5.020
    assert 51.274 < miny < 51.276
    assert 5.022 < maxx < 5.024
    assert 51.278 < maxy < 51.280


def test_load_geojson_centroid():
    """Centroid should be near (5.021, 51.277)."""
    lon, lat = load_geojson_centroid(TEST_FIELD_GEOJSON)
    assert lon == pytest.approx(5.021, abs=0.002)
    assert lat == pytest.approx(51.277, abs=0.003)


# ---------------------------------------------------------------------------
# Edge-case tests using temporary GeoJSON files
# ---------------------------------------------------------------------------


def test_load_geojson_empty_raises(tmp_path):
    """A GeoJSON with no coordinates should raise ValueError."""
    empty_geojson = tmp_path / "empty.geojson"
    empty_geojson.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [],
    }))
    with pytest.raises(ValueError, match="No coordinates"):
        load_geojson_bounds(str(empty_geojson))


def test_load_geojson_nested_features(tmp_path):
    """Features containing nested geometry should be parsed correctly."""
    nested = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[1.0, 10.0], [2.0, 10.0], [2.0, 11.0], [1.0, 11.0], [1.0, 10.0]]
                    ],
                },
            },
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[3.0, 12.0], [4.0, 12.0], [4.0, 13.0], [3.0, 13.0], [3.0, 12.0]]
                    ],
                },
            },
        ],
    }
    path = tmp_path / "nested.geojson"
    path.write_text(json.dumps(nested))

    minx, miny, maxx, maxy = load_geojson_bounds(str(path))
    assert minx == pytest.approx(1.0)
    assert miny == pytest.approx(10.0)
    assert maxx == pytest.approx(4.0)
    assert maxy == pytest.approx(13.0)


def test_load_geojson_multipolygon(tmp_path):
    """MultiPolygon geometries should be handled correctly."""
    multi = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
                [[[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 6.0], [5.0, 5.0]]],
            ],
        },
    }
    path = tmp_path / "multi.geojson"
    path.write_text(json.dumps(multi))

    minx, miny, maxx, maxy = load_geojson_bounds(str(path))
    assert minx == pytest.approx(0.0)
    assert miny == pytest.approx(0.0)
    assert maxx == pytest.approx(6.0)
    assert maxy == pytest.approx(6.0)


def test_load_geojson_nonexistent_raises():
    """A path that does not exist should raise FileNotFoundError or similar."""
    with pytest.raises((FileNotFoundError, OSError)):
        load_geojson_bounds("/nonexistent/path/field.geojson")
