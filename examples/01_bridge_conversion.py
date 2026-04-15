"""Example 01: Bridge conversion from ARC arrays to SCOPE format.

This script demonstrates how the bridge module converts ARC-format arrays
into SCOPE-ready xarray DataArrays.  It creates synthetic ARC outputs
programmatically, so no external files or dependencies beyond the core
package are needed.

Requirements:
    pip install arc-scope
"""

from __future__ import annotations

import numpy as np

from arc_scope.bridge import arc_arrays_to_scope_inputs
from arc_scope.bridge.parameter_map import (
    ARC_BIO_INDICES,
    ARC_BIO_NAMES,
    ARC_SOIL_INDICES,
    BIO_BANDS,
    BIO_SCALES,
    SCALE_BANDS,
)
from arc_scope.bridge.soil import validate_soil_params
from arc_scope.data import TEST_FIELD_GEOJSON


def main() -> None:
    print("=" * 60)
    print("ARC-SCOPE Bridge Conversion Example")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Understand the ARC output format
    # ------------------------------------------------------------------
    print("\n--- Step 1: ARC output format ---")
    print(f"BIO_BANDS (7 biophysical params): {BIO_BANDS}")
    print(f"SCALE_BANDS (15 total):           {SCALE_BANDS}")
    print(f"ARC_BIO_NAMES:                    {ARC_BIO_NAMES}")
    print(f"\nBIO_SCALES (integer -> physical):")
    for i, (name, scale) in enumerate(zip(BIO_BANDS, BIO_SCALES)):
        print(f"  [{i}] {name:>7s}  x {scale}")

    print(f"\nARC biophysical index mapping:")
    for idx, (scope_name, scale) in ARC_BIO_INDICES.items():
        print(f"  ARC[{idx}] -> SCOPE '{scope_name}' (scale {scale})")

    print(f"\nARC soil index mapping:")
    for idx, scope_name in ARC_SOIL_INDICES.items():
        print(f"  ARC[{idx}] -> SCOPE '{scope_name}'")

    # ------------------------------------------------------------------
    # Step 2: Create synthetic ARC outputs
    # ------------------------------------------------------------------
    print("\n--- Step 2: Create synthetic ARC outputs ---")

    ny, nx = 10, 10
    nt = 6

    # Boolean mask: True = masked OUT (invalid pixel)
    mask = np.zeros((ny, nx), dtype=bool)
    mask[0, :] = True   # mask out first row
    mask[-1, :] = True   # mask out last row
    n_valid = int((~mask).sum())
    print(f"Grid: {ny}x{nx} pixels, {n_valid} valid, {nt} timesteps")

    rng = np.random.default_rng(42)

    # post_bio_tensor: integer-coded biophysical parameters
    # Shape: (n_valid_pixels, 7_bands, n_times)
    post_bio_tensor = rng.integers(50, 500, size=(n_valid, 7, nt)).astype(np.float64)
    print(f"post_bio_tensor shape: {post_bio_tensor.shape}")

    # scale_data: bio scales + phenology + soil
    # Shape: (n_valid_pixels, 15)
    scale_data = np.zeros((n_valid, 15), dtype=np.float64)
    scale_data[:, :7] = rng.random((n_valid, 7)) * 50       # bio scales
    scale_data[:, 7:11] = rng.random((n_valid, 4)) * 10     # phenology
    scale_data[:, 11] = rng.uniform(0.1, 0.7, n_valid)      # BSMBrightness
    scale_data[:, 12] = rng.uniform(10.0, 30.0, n_valid)    # BSMlat
    scale_data[:, 13] = rng.uniform(10.0, 70.0, n_valid)    # BSMlon
    scale_data[:, 14] = rng.uniform(2.0, 100.0, n_valid)    # SMC
    print(f"scale_data shape:      {scale_data.shape}")

    # Day-of-year values (June-July 2021)
    doys = np.array([150, 160, 170, 180, 190, 200])

    # GDAL-style geotransform: [x_origin, x_res, x_rot, y_origin, y_rot, y_res]
    # Places the grid near the bundled test field in Belgium
    geotransform = np.array([5.019, 0.001, 0.0, 51.278, 0.0, -0.001])

    print(f"Bundled test field:    {TEST_FIELD_GEOJSON}")

    # ------------------------------------------------------------------
    # Step 3: Convert to SCOPE format using the bridge
    # ------------------------------------------------------------------
    print("\n--- Step 3: Bridge conversion ---")

    post_bio_da, post_bio_scale_da = arc_arrays_to_scope_inputs(
        post_bio_tensor=post_bio_tensor,
        scale_data=scale_data,
        mask=mask,
        doys=doys,
        geotransform=geotransform,
        crs="EPSG:4326",
        year=2021,
    )

    print(f"post_bio_da:")
    print(f"  dims:   {post_bio_da.dims}")
    print(f"  shape:  {post_bio_da.shape}")
    print(f"  bands:  {list(post_bio_da.coords['band'].values)}")
    print(f"  times:  {list(post_bio_da.coords['time'].values[:3])}...")
    print(f"  y range: [{post_bio_da.coords['y'].values[0]:.3f}, "
          f"{post_bio_da.coords['y'].values[-1]:.3f}]")
    print(f"  x range: [{post_bio_da.coords['x'].values[0]:.3f}, "
          f"{post_bio_da.coords['x'].values[-1]:.3f}]")

    print(f"\npost_bio_scale_da:")
    print(f"  dims:   {post_bio_scale_da.dims}")
    print(f"  shape:  {post_bio_scale_da.shape}")
    print(f"  bands:  {list(post_bio_scale_da.coords['band'].values)}")

    # ------------------------------------------------------------------
    # Step 4: Inspect the converted values
    # ------------------------------------------------------------------
    print("\n--- Step 4: Inspect converted values ---")
    print("Sample physical values at pixel (1, 0), first timestep:")

    for i, band in enumerate(BIO_BANDS):
        raw = post_bio_tensor[0, i, 0]
        scaled = post_bio_da.values[1, 0, i, 0]  # row 1 because row 0 is masked
        print(f"  {band:>7s}: raw={raw:8.1f}  scaled={scaled:10.6f}  "
              f"(x {BIO_SCALES[i]})")

    # ------------------------------------------------------------------
    # Step 5: Validate soil parameters
    # ------------------------------------------------------------------
    print("\n--- Step 5: Soil parameter validation ---")

    validated = validate_soil_params(
        brightness=scale_data[:, 11],
        lat=scale_data[:, 12],
        lon=scale_data[:, 13],
        smc=scale_data[:, 14],
    )
    for name, arr in validated.items():
        print(f"  {name:>15s}: min={arr.min():.3f}  max={arr.max():.3f}")

    print("\nBridge conversion complete.")
    print("The output DataArrays are ready for prepare_scope_input_dataset().")


if __name__ == "__main__":
    main()
