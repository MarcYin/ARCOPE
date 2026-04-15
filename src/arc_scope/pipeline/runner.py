"""High-level pipeline runner orchestrating ARC -> Bridge -> Weather -> SCOPE."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xarray as xr

from arc_scope.pipeline.config import PipelineConfig
from arc_scope.pipeline.steps import (
    ArcResult,
    bridge_arc_to_scope,
    build_observation_dataset,
    fetch_weather,
    prepare_scope_dataset,
    retrieve_arc,
    run_scope_simulation,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Container for full pipeline results."""

    arc_result: ArcResult | None = None
    post_bio_da: xr.DataArray | None = None
    post_bio_scale_da: xr.DataArray | None = None
    weather_ds: xr.Dataset | None = None
    observation_ds: xr.Dataset | None = None
    scope_input_ds: xr.Dataset | None = None
    scope_output_ds: xr.Dataset | None = None


class ArcScopePipeline:
    """End-to-end pipeline from field definition to SCOPE simulation.

    Usage::

        config = PipelineConfig(
            geojson_path="field.geojson",
            start_date="2021-05-15",
            end_date="2021-10-01",
            crop_type="wheat",
            start_of_season=170,
            year=2021,
            scope_workflow="fluorescence",
        )
        pipeline = ArcScopePipeline(config)
        result = pipeline.run()

    Individual steps can also be run separately::

        arc_result = pipeline.run_arc()
        bio_da, scale_da = pipeline.run_bridge(arc_result)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self) -> PipelineResult:
        """Execute the full pipeline: ARC -> Bridge -> Weather -> SCOPE.

        Returns
        -------
        PipelineResult with all intermediate and final outputs.
        """
        result = PipelineResult()

        # Step 1: ARC retrieval
        logger.info("Running ARC retrieval...")
        result.arc_result = self.run_arc()

        # Step 2: Bridge ARC -> SCOPE format
        logger.info("Bridging ARC outputs to SCOPE format...")
        result.post_bio_da, result.post_bio_scale_da = self.run_bridge(result.arc_result)

        # Step 3: Fetch weather data (skip for reflectance-only)
        if self.config.scope_workflow != "reflectance":
            logger.info("Fetching weather data...")
            result.weather_ds = self.run_weather()
        else:
            logger.info("Reflectance workflow: creating minimal weather dataset...")
            result.weather_ds = self._minimal_weather(result.arc_result)

        # Step 4: Build observation geometry
        logger.info("Building observation geometry...")
        result.observation_ds = self.run_observation(result.arc_result)

        # Step 5: Prepare SCOPE dataset
        logger.info("Preparing SCOPE input dataset...")
        result.scope_input_ds = prepare_scope_dataset(
            result.post_bio_da,
            result.post_bio_scale_da,
            result.weather_ds,
            result.observation_ds,
            self.config,
        )

        # Step 6: Run SCOPE simulation
        logger.info("Running SCOPE simulation...")
        result.scope_output_ds = run_scope_simulation(
            result.scope_input_ds,
            self.config,
        )

        # Step 7: Save outputs
        if self.config.save_scope_netcdf:
            self._save_scope_output(result.scope_output_ds)

        logger.info("Pipeline complete.")
        return result

    def run_arc(self) -> ArcResult:
        """Run ARC retrieval step only."""
        return retrieve_arc(self.config)

    def run_bridge(
        self,
        arc_result: ArcResult,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Convert ARC outputs to SCOPE format."""
        return bridge_arc_to_scope(arc_result, self.config.year)

    def run_weather(self) -> xr.Dataset:
        """Fetch weather data for the configured field and time range."""
        return fetch_weather(self.config)

    def run_observation(self, arc_result: ArcResult) -> xr.Dataset:
        """Build the observation geometry dataset."""
        return build_observation_dataset(
            doys=arc_result.doys,
            year=self.config.year,
            geojson_path=self.config.geojson_path,
        )

    def run_scope(
        self,
        post_bio_da: xr.DataArray,
        post_bio_scale_da: xr.DataArray,
        weather_ds: xr.Dataset,
        observation_ds: xr.Dataset,
    ) -> xr.Dataset:
        """Prepare and run SCOPE from bridge outputs + weather + observations."""
        scope_ds = prepare_scope_dataset(
            post_bio_da, post_bio_scale_da,
            weather_ds, observation_ds,
            self.config,
        )
        return run_scope_simulation(scope_ds, self.config)

    def _minimal_weather(self, arc_result: ArcResult) -> xr.Dataset:
        """Create a minimal weather dataset for reflectance-only simulations.

        Reflectance workflow doesn't actually need weather, but
        ``prepare_scope_input_dataset`` expects a weather dataset.
        This creates one with placeholder values.
        """
        import numpy as np
        import pandas as pd

        times = pd.to_datetime(
            [f"{self.config.year}{int(d):03d}" for d in arc_result.doys],
            format="%Y%j",
        )

        n = len(times)
        return xr.Dataset(
            {
                "Rin": ("time", np.full(n, 500.0)),
                "Rli": ("time", np.full(n, 300.0)),
                "Ta": ("time", np.full(n, 20.0)),
                "ea": ("time", np.full(n, 15.0)),
                "p": ("time", np.full(n, 1013.0)),
                "u": ("time", np.full(n, 2.0)),
            },
            coords={"time": times},
        )

    def _save_scope_output(self, ds: xr.Dataset) -> Path:
        """Save SCOPE output dataset to NetCDF."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.config.output_dir / "scope_output.nc"

        try:
            from scope.io.export import write_netcdf_dataset

            write_netcdf_dataset(ds, output_path)
        except ImportError:
            ds.to_netcdf(output_path)

        logger.info("SCOPE output saved to %s", output_path)
        return output_path
