"""Example 02: Preparing a SCOPE-ready reflectance dataset.

This script shows what a prepared SCOPE input dataset looks like, creates
a synthetic one for inspection, and optionally runs a SCOPE reflectance
simulation if scope-rtm is installed.

Requirements:
    pip install arc-scope               # for dataset inspection
    pip install "arc-scope[scope]"      # to actually run SCOPE
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from arc_scope.bridge import arc_arrays_to_scope_inputs
from arc_scope.bridge.parameter_map import BIO_BANDS, SCALE_BANDS
from arc_scope.pipeline.config import WORKFLOW_OPTIONS


def create_synthetic_scope_dataset() -> tuple[xr.DataArray, xr.DataArray, xr.Dataset, xr.Dataset]:
    """Create synthetic bridge outputs, weather, and observation datasets."""

    # --- Bridge outputs ---
    ny, nx, nt = 4, 4, 3
    mask = np.zeros((ny, nx), dtype=bool)
    n_valid = ny * nx

    rng = np.random.default_rng(123)
    post_bio_tensor = rng.integers(100, 400, size=(n_valid, 7, nt)).astype(np.float64)
    scale_data = np.zeros((n_valid, 15), dtype=np.float64)
    scale_data[:, :7] = rng.random((n_valid, 7)) * 50
    scale_data[:, 7:11] = rng.random((n_valid, 4)) * 10
    scale_data[:, 11] = rng.uniform(0.2, 0.5, n_valid)    # BSMBrightness
    scale_data[:, 12] = rng.uniform(15.0, 25.0, n_valid)  # BSMlat
    scale_data[:, 13] = rng.uniform(20.0, 50.0, n_valid)  # BSMlon
    scale_data[:, 14] = rng.uniform(10.0, 60.0, n_valid)  # SMC

    doys = np.array([170, 180, 190])
    geotransform = np.array([5.02, 0.001, 0.0, 51.278, 0.0, -0.001])

    post_bio_da, post_bio_scale_da = arc_arrays_to_scope_inputs(
        post_bio_tensor=post_bio_tensor,
        scale_data=scale_data,
        mask=mask,
        doys=doys,
        geotransform=geotransform,
        crs="EPSG:4326",
        year=2021,
    )

    # --- Weather dataset (minimal for reflectance workflow) ---
    times = pd.to_datetime([f"2021{int(d):03d}" for d in doys], format="%Y%j")
    weather_ds = xr.Dataset(
        {
            "Rin": ("time", np.array([500.0, 600.0, 450.0])),
            "Rli": ("time", np.array([300.0, 310.0, 290.0])),
            "Ta": ("time", np.array([22.0, 25.0, 20.0])),
            "ea": ("time", np.array([15.0, 18.0, 14.0])),
            "p": ("time", np.array([1013.0, 1010.0, 1015.0])),
            "u": ("time", np.array([2.0, 3.0, 1.5])),
        },
        coords={"time": times},
    )

    # --- Observation geometry ---
    observation_ds = xr.Dataset(
        {
            "solar_zenith_angle": ("time", np.array([30.0, 28.0, 32.0])),
            "viewing_zenith_angle": ("time", np.array([0.0, 0.0, 0.0])),
            "solar_azimuth_angle": ("time", np.array([150.0, 155.0, 148.0])),
            "viewing_azimuth_angle": ("time", np.array([0.0, 0.0, 0.0])),
            "delta_time": ("time", times.values),
        },
        coords={"time": times},
    )

    return post_bio_da, post_bio_scale_da, weather_ds, observation_ds


def main() -> None:
    print("=" * 60)
    print("ARC-SCOPE Reflectance Simulation Example")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Review SCOPE workflow options
    # ------------------------------------------------------------------
    print("\n--- Step 1: SCOPE workflow options ---")
    for name, opts in WORKFLOW_OPTIONS.items():
        print(f"  {name:>16s}: {opts}")

    # ------------------------------------------------------------------
    # Step 2: Create synthetic datasets
    # ------------------------------------------------------------------
    print("\n--- Step 2: Create synthetic SCOPE-ready datasets ---")
    post_bio_da, post_bio_scale_da, weather_ds, observation_ds = (
        create_synthetic_scope_dataset()
    )

    print(f"\npost_bio_da:")
    print(f"  shape: {post_bio_da.shape}  dims: {post_bio_da.dims}")
    print(f"  bands: {list(post_bio_da.coords['band'].values)}")

    print(f"\npost_bio_scale_da:")
    print(f"  shape: {post_bio_scale_da.shape}  dims: {post_bio_scale_da.dims}")

    print(f"\nweather_ds variables: {list(weather_ds.data_vars)}")
    print(f"observation_ds variables: {list(observation_ds.data_vars)}")

    # ------------------------------------------------------------------
    # Step 3: Show dataset structure expected by SCOPE
    # ------------------------------------------------------------------
    print("\n--- Step 3: Expected SCOPE input structure ---")
    print("SCOPE's prepare_scope_input_dataset() merges these four datasets")
    print("into a single xr.Dataset with variables including:")
    print("  - Biophysical: N, Cab, Cdm, Cw, LAI, ala, Cs")
    print("  - Soil BSM: BSMBrightness, BSMlat, BSMlon, SMC")
    print("  - Weather: Rin, Rli, Ta, ea, p, u")
    print("  - Geometry: solar_zenith_angle, viewing_zenith_angle, ...")
    print("  - Spectral: Esun_sw, Esky_sw (from radiation partitioning)")

    # ------------------------------------------------------------------
    # Step 4: Sample values from the bio DataArray
    # ------------------------------------------------------------------
    print("\n--- Step 4: Sample biophysical values ---")
    print("Physical values at pixel (0, 0) across time:")
    for band in BIO_BANDS:
        vals = post_bio_da.sel(band=band).values[0, 0, :]
        print(f"  {band:>7s}: {np.array2string(vals, precision=4, separator=', ')}")

    # ------------------------------------------------------------------
    # Step 5: Try running SCOPE (if installed)
    # ------------------------------------------------------------------
    print("\n--- Step 5: SCOPE simulation ---")
    try:
        from scope.io.prepare import prepare_scope_input_dataset  # noqa: F401

        print("scope-rtm is installed. In a real workflow you would call:")
        print("  scope_ds = prepare_scope_dataset(bio_da, scale_da, weather, obs, config)")
        print("  output = run_scope_simulation(scope_ds, config)")
        print("See examples/03_full_pipeline.py for the complete workflow.")
    except ImportError:
        print("scope-rtm is not installed.")
        print("To run SCOPE simulations, install with:")
        print("  pip install \"arc-scope[scope]\"")

    print("\nDone.")


if __name__ == "__main__":
    main()
