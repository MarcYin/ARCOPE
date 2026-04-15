"""Tests for bridge parameter mapping constants."""

from arc_scope.bridge.parameter_map import (
    ARC_BIO_INDICES,
    BIO_BANDS,
    BIO_SCALES,
    SCALE_BANDS,
)


def test_bio_bands_count():
    assert len(BIO_BANDS) == 7


def test_bio_scales_count():
    assert len(BIO_SCALES) == 7


def test_scale_bands_count():
    assert len(SCALE_BANDS) == 15


def test_arc_bio_indices_match_bio_bands():
    """ARC_BIO_INDICES should map to all 7 biophysical parameters."""
    assert len(ARC_BIO_INDICES) == 7
    for i in range(7):
        assert i in ARC_BIO_INDICES


def test_scale_bands_starts_with_bio():
    """First 7 SCALE_BANDS should match BIO_BANDS."""
    for i in range(7):
        assert SCALE_BANDS[i] == BIO_BANDS[i]


def test_soil_bands_in_scale():
    """Soil bands should be at indices 11-14."""
    assert SCALE_BANDS[11] == "BSMBrightness"
    assert SCALE_BANDS[12] == "BSMlat"
    assert SCALE_BANDS[13] == "BSMlon"
    assert SCALE_BANDS[14] == "SMC"
