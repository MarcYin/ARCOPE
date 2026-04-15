"""Mapping constants between ARC biophysical parameter indices and SCOPE variable names.

ARC's ``arc_field()`` returns ``post_bio_tensor`` with 7 biophysical parameters
per pixel per timestep, and ``scale_data`` with 15 columns (7 bio scales + 4
phenology + 4 soil).  SCOPE expects these as named xarray variables with
specific units and scale factors.

The constants here are derived from:
- ARC: ``arc_sample_generator.py`` (scale_samples, generate_arc_refs)
- SCOPE: ``src/scope/io/prepare.py`` (BIO_BANDS, BIO_SCALES, SCALE_BANDS)
"""

from __future__ import annotations

# SCOPE band names (lowercase) in the order stored in post_bio_tensor
BIO_BANDS: tuple[str, ...] = ("N", "cab", "cm", "cw", "lai", "ala", "cbrown")

# Scale factors applied to integer-coded ARC values to recover physical units.
# post_bio_da = post_bio_tensor * BIO_SCALES
BIO_SCALES: tuple[float, ...] = (
    1 / 100.0,    # N          (dimensionless)
    1 / 100.0,    # Cab        (ug cm-2)
    1 / 10000.0,  # Cm -> Cdm  (g cm-2)
    1 / 10000.0,  # Cw         (g cm-2)
    1 / 100.0,    # LAI        (m2 m-2)
    1 / 100.0,    # ala        (deg)
    1 / 1000.0,   # Cbrown     (dimensionless)
)

# Full 15-band names for the scale_data array (bio + phenology + soil)
SCALE_BANDS: tuple[str, ...] = (
    "N", "cab", "cm", "cw", "lai", "ala", "cbrown",  # bio (0-6)
    "n0", "m0", "n1", "m1",                           # phenology (7-10)
    "BSMBrightness", "BSMlat", "BSMlon", "SMC",       # soil (11-14)
)

# Mapping from ARC post_bio_tensor column index to (SCOPE variable name, scale factor)
ARC_BIO_INDICES: dict[int, tuple[str, float]] = {
    0: ("N",       1 / 100.0),
    1: ("Cab",     1 / 100.0),
    2: ("Cdm",     1 / 10000.0),   # ARC calls this "Cm"
    3: ("Cw",      1 / 10000.0),
    4: ("LAI",     1 / 100.0),
    5: ("ala",     1 / 100.0),
    6: ("Cs",      1 / 1000.0),    # ARC calls this "Cbrown"
}

# Human-readable ARC parameter names in tensor column order
ARC_BIO_NAMES: tuple[str, ...] = ("N", "Cab", "Cm", "Cw", "LAI", "ALA", "Cbrown")

# Mapping from scale_data column index to SCOPE soil variable name
ARC_SOIL_INDICES: dict[int, str] = {
    11: "BSMBrightness",
    12: "BSMlat",
    13: "BSMlon",
    14: "SMC",
}

# Physical ranges for ARC biophysical parameters (from generate_arc_refs)
ARC_BIO_RANGES: dict[str, tuple[float, float]] = {
    "N":       (1.0,   3.0),
    "Cab":     (20.0,  80.0),
    "Cm":      (0.001, 0.04),
    "Cw":      (0.001, 0.1),
    "LAI":     (0.5,   8.0),
    "ALA":     (50.0,  80.0),
    "Cbrown":  (0.0,   1.5),
}

# Physical ranges for ARC soil parameters (from generate_arc_refs)
ARC_SOIL_RANGES: dict[str, tuple[float, float]] = {
    "BSMBrightness":  (0.1, 0.7),
    "BSMlat":         (10.0, 30.0),
    "BSMlon":         (10.0, 70.0),
    "SMC":            (2.0, 100.0),
}
