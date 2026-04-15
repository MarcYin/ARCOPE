"""Tests for ARC-to-SCOPE data conversion."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from arc_scope.bridge.convert import arc_arrays_to_scope_inputs, arc_npz_to_scope_inputs
from arc_scope.bridge.parameter_map import BIO_BANDS, BIO_SCALES, SCALE_BANDS


def test_arc_arrays_to_scope_inputs_shapes(sample_arc_outputs):
    """Output DataArrays should have correct dimensions and shapes."""
    out = sample_arc_outputs
    bio_da, scale_da = arc_arrays_to_scope_inputs(
        post_bio_tensor=out["post_bio_tensor"],
        scale_data=out["scale_data"],
        mask=out["mask"],
        doys=out["doys"],
        geotransform=out["geotransform"],
        crs=out["crs"],
        year=2021,
    )

    assert bio_da.dims == ("y", "x", "band", "time")
    assert scale_da.dims == ("y", "x", "band")

    ny, nx = out["mask"].shape
    assert bio_da.sizes["y"] == ny
    assert bio_da.sizes["x"] == nx
    assert bio_da.sizes["band"] == len(BIO_BANDS)
    assert scale_da.sizes["band"] == len(SCALE_BANDS)


def test_masked_pixels_are_nan(sample_arc_outputs):
    """Masked pixels should be NaN in the output."""
    out = sample_arc_outputs
    bio_da, scale_da = arc_arrays_to_scope_inputs(
        post_bio_tensor=out["post_bio_tensor"],
        scale_data=out["scale_data"],
        mask=out["mask"],
        doys=out["doys"],
        geotransform=out["geotransform"],
        crs=out["crs"],
        year=2021,
    )

    # First row is masked in sample_mask
    assert np.all(np.isnan(bio_da.values[0, :, :, :]))
    # Non-masked rows should have finite values
    assert np.all(np.isfinite(bio_da.values[1, :, :, :]))


def test_scale_factors_applied(sample_arc_outputs):
    """BIO_SCALES should be applied to convert integer-coded values."""
    out = sample_arc_outputs
    bio_da, _ = arc_arrays_to_scope_inputs(
        post_bio_tensor=out["post_bio_tensor"],
        scale_data=out["scale_data"],
        mask=out["mask"],
        doys=out["doys"],
        geotransform=out["geotransform"],
        crs=out["crs"],
        year=2021,
    )

    # The maximum integer-coded value is ~1000.
    # With scale factors (max 1/100), physical values should be ~10 at most.
    valid = bio_da.values[np.isfinite(bio_da.values)]
    assert valid.max() < 1000 * max(BIO_SCALES) + 1


def test_coordinates_from_geotransform(sample_arc_outputs):
    """Spatial coordinates should be derived from geotransform."""
    out = sample_arc_outputs
    bio_da, _ = arc_arrays_to_scope_inputs(
        post_bio_tensor=out["post_bio_tensor"],
        scale_data=out["scale_data"],
        mask=out["mask"],
        doys=out["doys"],
        geotransform=out["geotransform"],
        crs=out["crs"],
        year=2021,
    )

    gt = out["geotransform"]
    assert bio_da.coords["x"].values[0] == pytest.approx(gt[0])
    assert bio_da.coords["y"].values[0] == pytest.approx(gt[3])


def test_time_coordinates(sample_arc_outputs):
    """Time coordinates should be datetime values for the given year."""
    out = sample_arc_outputs
    bio_da, _ = arc_arrays_to_scope_inputs(
        post_bio_tensor=out["post_bio_tensor"],
        scale_data=out["scale_data"],
        mask=out["mask"],
        doys=out["doys"],
        geotransform=out["geotransform"],
        crs=out["crs"],
        year=2021,
    )

    times = bio_da.coords["time"].values
    assert len(times) == len(out["doys"])
    # Check that all times are in 2021
    for t in times:
        assert str(t)[:4] == "2021"


# ---------------------------------------------------------------------------
# NPZ roundtrip tests
# ---------------------------------------------------------------------------


def test_npz_roundtrip(tmp_npz_path):
    """Save synthetic NPZ in ARC format, reload via arc_npz_to_scope_inputs,
    and verify shapes match expectations."""
    bio_da, scale_da = arc_npz_to_scope_inputs(tmp_npz_path, year=2021)

    assert bio_da.dims == ("y", "x", "band", "time")
    assert scale_da.dims == ("y", "x", "band")
    assert bio_da.sizes["band"] == len(BIO_BANDS)
    assert scale_da.sizes["band"] == len(SCALE_BANDS)
    # The mask is 10x10 with first and last rows masked
    assert bio_da.sizes["y"] == 10
    assert bio_da.sizes["x"] == 10


def test_single_pixel_field():
    """A mask with only 1 valid pixel should still produce valid output."""
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False  # single valid pixel

    n_valid = 1
    n_times = 3
    rng = np.random.default_rng(0)
    post_bio_tensor = rng.random((n_valid, 7, n_times)) * 100
    scale_data = rng.random((n_valid, 15)) * 50
    doys = np.array([100, 150, 200])
    geotransform = np.array([5.0, 0.001, 0.0, 51.0, 0.0, -0.001])

    bio_da, scale_da = arc_arrays_to_scope_inputs(
        post_bio_tensor=post_bio_tensor,
        scale_data=scale_data,
        mask=mask,
        doys=doys,
        geotransform=geotransform,
        crs="EPSG:4326",
        year=2021,
    )

    # Only pixel (2,2) should have finite values
    assert np.isfinite(bio_da.values[2, 2, 0, 0])
    assert np.all(np.isnan(bio_da.values[0, 0, :, :]))
    assert bio_da.sizes["time"] == n_times


def test_duplicate_doys_deduplicated():
    """Duplicate DOY entries should be collapsed to unique times."""
    mask = np.zeros((3, 3), dtype=bool)
    n_valid = 9
    # DOYs with a duplicate
    doys = np.array([150, 160, 160, 170])
    n_times = len(doys)

    rng = np.random.default_rng(1)
    post_bio_tensor = rng.random((n_valid, 7, n_times)) * 100
    scale_data = rng.random((n_valid, 15)) * 50
    geotransform = np.array([5.0, 0.001, 0.0, 51.0, 0.0, -0.001])

    bio_da, _ = arc_arrays_to_scope_inputs(
        post_bio_tensor=post_bio_tensor,
        scale_data=scale_data,
        mask=mask,
        doys=doys,
        geotransform=geotransform,
        crs="EPSG:4326",
        year=2021,
    )

    # After deduplication, should have 3 unique times (150, 160, 170)
    assert bio_da.sizes["time"] == 3
