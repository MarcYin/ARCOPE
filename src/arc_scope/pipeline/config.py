"""Pipeline configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arc_scope.utils.types import PathLike


# Map user-friendly workflow names to SCOPE option flags
WORKFLOW_OPTIONS: dict[str, dict[str, int]] = {
    "reflectance": {
        "calc_fluor": 0,
        "calc_planck": 0,
    },
    "fluorescence": {
        "calc_fluor": 1,
        "calc_planck": 0,
    },
    "thermal": {
        "calc_fluor": 0,
        "calc_planck": 1,
    },
    "energy-balance": {
        "calc_fluor": 1,
        "calc_planck": 1,
    },
}


@dataclass
class PipelineConfig:
    """Master configuration for the ARC-SCOPE pipeline.

    Parameters
    ----------
    geojson_path:
        Path to GeoJSON file defining the field boundary.
    start_date:
        Start date for satellite data, e.g. ``"2021-05-15"``.
    end_date:
        End date for satellite data.
    crop_type:
        Crop type identifier, e.g. ``"wheat"``, ``"maize"``.
    start_of_season:
        Day of year when the growth season begins.
    year:
        Calendar year for the simulation period.

    ARC options
    -----------
    num_samples:
        Number of archetype samples to generate (default 100000).
    growth_season_length:
        Length of growth season in days (default 45).
    s2_data_folder:
        Directory for caching Sentinel-2 downloads.
    data_source:
        S2 data source: ``"aws"``, ``"planetary"``, ``"cdse"``, ``"gee"``.

    Weather options
    ---------------
    weather_provider:
        Provider name: ``"era5"`` or ``"local"``.
    weather_config:
        Provider-specific configuration dict.

    SCOPE options
    -------------
    scope_workflow:
        Simulation workflow: ``"reflectance"``, ``"fluorescence"``,
        ``"thermal"``, or ``"energy-balance"``.
    scope_root_path:
        Path to SCOPE upstream assets directory.
    scope_options:
        Additional SCOPE options to override defaults.
    device:
        PyTorch device for SCOPE (``"cpu"`` or ``"cuda"``).
    dtype:
        PyTorch dtype string (``"float32"`` or ``"float64"``).

    Output options
    --------------
    output_dir:
        Directory for saving results.
    save_arc_npz:
        Whether to save ARC outputs to NPZ.
    save_scope_netcdf:
        Whether to save SCOPE outputs to NetCDF.

    Optimization options
    --------------------
    optimize:
        Enable parameter optimization loop.
    optim_config:
        Optimization-specific configuration.
    """

    # Field definition (required)
    geojson_path: PathLike
    start_date: str
    end_date: str
    crop_type: str
    start_of_season: int
    year: int

    # ARC options
    num_samples: int = 100000
    growth_season_length: int = 45
    s2_data_folder: PathLike | None = None
    data_source: str = "aws"

    # Weather options
    weather_provider: str = "era5"
    weather_config: dict[str, Any] = field(default_factory=dict)

    # SCOPE options
    scope_workflow: str = "reflectance"
    scope_root_path: PathLike | None = None
    scope_options: dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"
    dtype: str = "float64"

    # Output options
    output_dir: PathLike = Path("./output")
    save_arc_npz: bool = True
    save_scope_netcdf: bool = True

    # Optimization (future)
    optimize: bool = False
    optim_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.geojson_path = Path(self.geojson_path)
        self.output_dir = Path(self.output_dir)
        if self.s2_data_folder is not None:
            self.s2_data_folder = Path(self.s2_data_folder)
        if self.scope_workflow not in WORKFLOW_OPTIONS:
            raise ValueError(
                f"Unknown workflow '{self.scope_workflow}'. "
                f"Choose from: {list(WORKFLOW_OPTIONS.keys())}"
            )

    @property
    def resolved_scope_options(self) -> dict[str, Any]:
        """Merge workflow defaults with user overrides."""
        opts = dict(WORKFLOW_OPTIONS[self.scope_workflow])
        opts.update(self.scope_options)
        return opts
