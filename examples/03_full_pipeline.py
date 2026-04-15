"""Example 03: Full ARC-SCOPE pipeline configuration.

This script shows the complete PipelineConfig setup with detailed
comments on each field, demonstrates how to use local weather files,
and shows how to run individual pipeline steps.

Requirements:
    pip install "arc-scope[all]"    # for the full pipeline

The script handles ImportError gracefully so you can read and understand
the configuration even without all dependencies installed.
"""

from __future__ import annotations

from pathlib import Path

from arc_scope.data import TEST_FIELD_GEOJSON
from arc_scope.pipeline.config import PipelineConfig, WORKFLOW_OPTIONS


def show_era5_config() -> PipelineConfig:
    """Build a PipelineConfig using ERA5 weather data."""
    config = PipelineConfig(
        # --- Field definition (required) ---
        # GeoJSON polygon defining the field boundary.
        # The bundled test field is in Belgium (Flanders).
        geojson_path=TEST_FIELD_GEOJSON,

        # Sentinel-2 date range for satellite imagery search.
        start_date="2021-05-15",
        end_date="2021-10-01",

        # Crop type drives the prior distributions in ARC's
        # Bayesian retrieval (e.g., plausible LAI ranges differ
        # between wheat and maize).
        crop_type="wheat",

        # Day of year when the growth season begins.
        # Used by ARC to anchor the phenology model.
        start_of_season=170,

        # Calendar year (combined with DOYs to build datetime coords).
        year=2021,

        # --- ARC retrieval options ---
        # Number of archetype samples drawn for the prior.
        # Higher = more accurate but slower.
        num_samples=100000,

        # Growth season length in days, controls temporal smoothness.
        growth_season_length=60,

        # Where to cache downloaded Sentinel-2 tiles.
        s2_data_folder=Path("./S2_data"),

        # Data source for Sentinel-2 imagery.
        # Options: "aws", "planetary", "cdse", "gee"
        data_source="aws",

        # --- Weather ---
        # ERA5 reanalysis: no extra config needed beyond ~/.cdsapirc
        weather_provider="era5",
        weather_config={},

        # --- SCOPE simulation ---
        # Workflow determines which outputs SCOPE computes:
        #   "reflectance"     -> directional reflectance only (fastest)
        #   "fluorescence"    -> reflectance + SIF (F685, F740)
        #   "thermal"         -> reflectance + thermal emission (LST)
        #   "energy-balance"  -> full energy balance (SIF + thermal + fluxes)
        scope_workflow="fluorescence",

        # Path to SCOPE assets directory (spectral databases, etc.)
        # Set to None to use the scope-rtm package defaults.
        scope_root_path=None,

        # Additional SCOPE option overrides (merged with workflow defaults).
        scope_options={},

        # PyTorch device and dtype for SCOPE runner.
        device="cpu",      # or "cuda" for GPU
        dtype="float64",   # or "float32" for faster but less precise

        # --- Output ---
        output_dir=Path("./output"),
        save_arc_npz=True,
        save_scope_netcdf=True,
    )
    return config


def show_local_weather_config() -> PipelineConfig:
    """Build a PipelineConfig using a local weather CSV file."""
    config = PipelineConfig(
        geojson_path=TEST_FIELD_GEOJSON,
        start_date="2021-05-15",
        end_date="2021-10-01",
        crop_type="wheat",
        start_of_season=170,
        year=2021,

        # --- Local weather provider ---
        weather_provider="local",
        weather_config={
            # Path to the local weather file (CSV or NetCDF).
            "file_path": "weather_station.csv",

            # Mapping from file column names to SCOPE variable names.
            # The keys are YOUR column names; values are SCOPE's names.
            "var_map": {
                "air_temp_c": "Ta",          # Air temperature (degC)
                "sw_down_wm2": "Rin",        # Incoming shortwave (W m-2)
                "lw_down_wm2": "Rli",        # Incoming longwave (W m-2)
                "vapour_pressure_hpa": "ea",  # Vapor pressure (hPa)
                "pressure_hpa": "p",         # Air pressure (hPa)
                "wind_speed_ms": "u",        # Wind speed (m s-1)
            },

            # Name of the timestamp column in the CSV.
            "time_column": "timestamp",
        },

        scope_workflow="fluorescence",
    )
    return config


def main() -> None:
    print("=" * 60)
    print("ARC-SCOPE Full Pipeline Example")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Show configuration with ERA5
    # ------------------------------------------------------------------
    print("\n--- Step 1: ERA5 configuration ---")
    config = show_era5_config()
    print(f"  geojson_path:     {config.geojson_path}")
    print(f"  date range:       {config.start_date} to {config.end_date}")
    print(f"  crop_type:        {config.crop_type}")
    print(f"  year:             {config.year}")
    print(f"  scope_workflow:   {config.scope_workflow}")
    print(f"  resolved options: {config.resolved_scope_options}")
    print(f"  device:           {config.device}")

    # ------------------------------------------------------------------
    # Step 2: Show local weather configuration
    # ------------------------------------------------------------------
    print("\n--- Step 2: Local weather configuration ---")
    local_config = show_local_weather_config()
    print(f"  weather_provider: {local_config.weather_provider}")
    print(f"  weather_config:   {local_config.weather_config}")

    # ------------------------------------------------------------------
    # Step 3: Available workflows
    # ------------------------------------------------------------------
    print("\n--- Step 3: Available SCOPE workflows ---")
    for name, opts in WORKFLOW_OPTIONS.items():
        print(f"  {name:>16s}: calc_fluor={opts['calc_fluor']}, "
              f"calc_planck={opts['calc_planck']}")

    # ------------------------------------------------------------------
    # Step 4: Try running individual steps
    # ------------------------------------------------------------------
    print("\n--- Step 4: Running individual steps ---")

    # Step 4a: ARC retrieval
    try:
        from arc_scope.pipeline.steps import retrieve_arc

        print("\nARC is available. To run retrieval:")
        print("  arc_result = retrieve_arc(config)")
        print("  (requires Sentinel-2 data access and GDAL)")
    except ImportError:
        print("\nARC is not installed.")
        print("  Install with: pip install \"arc-scope[arc]\"")

    # Step 4b: Bridge (always available)
    print("\nBridge module is always available (core dependency).")
    print("  from arc_scope.bridge import arc_arrays_to_scope_inputs")
    print("  post_bio_da, scale_da = arc_arrays_to_scope_inputs(...)")

    # Step 4c: Weather
    try:
        from arc_scope.weather.era5 import ERA5Provider

        print("\nERA5Provider is available.")
        print("  provider = ERA5Provider()")
        print("  weather_ds = provider.fetch(bounds, time_range)")
        print("  (requires ~/.cdsapirc credentials)")
    except ImportError:
        print("\ncdsapi is not installed.")
        print("  Install with: pip install \"arc-scope[weather]\"")

    # Step 4d: SCOPE
    try:
        from scope import ScopeGridRunner  # noqa: F401

        print("\nSCOPE is available. To run simulation:")
        print("  from arc_scope.pipeline import ArcScopePipeline")
        print("  pipeline = ArcScopePipeline(config)")
        print("  result = pipeline.run()")
    except ImportError:
        print("\nscope-rtm is not installed.")
        print("  Install with: pip install \"arc-scope[scope]\"")

    # ------------------------------------------------------------------
    # Step 5: Full pipeline execution
    # ------------------------------------------------------------------
    print("\n--- Step 5: Full pipeline ---")
    try:
        from arc_scope.pipeline import ArcScopePipeline

        print("To run the full pipeline:")
        print("  pipeline = ArcScopePipeline(config)")
        print("  result = pipeline.run()")
        print("  result.scope_output_ds  # xr.Dataset with SCOPE outputs")

        # Only attempt to run if all deps are present
        try:
            from arc import arc_field  # noqa: F401
            from scope import ScopeGridRunner  # noqa: F401

            print("\nAll dependencies present. Running pipeline...")
            pipeline = ArcScopePipeline(config)
            result = pipeline.run()
            print(f"ARC DOYs: {result.arc_result.doys}")
            print(f"SCOPE output vars: {list(result.scope_output_ds.data_vars)}")
        except ImportError:
            print("\nNot all dependencies installed for a live run.")
            print("Install everything with: pip install \"arc-scope[all]\"")
        except Exception as exc:
            print(f"\nPipeline run failed (expected without real data): {exc}")

    except ImportError:
        print("Pipeline module requires: pip install \"arc-scope[all]\"")

    print("\nDone.")


if __name__ == "__main__":
    main()
