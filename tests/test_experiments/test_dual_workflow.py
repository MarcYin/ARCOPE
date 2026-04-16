"""Tests for the heavyweight ARC-SCOPE experiment helpers."""

from __future__ import annotations

import pandas as pd
import numpy as np
import xarray as xr

from arc_scope.experiments.dual_workflow import (
    DualWorkflowExperimentResult,
    RuntimeCheck,
    WorkflowRun,
    _reduce_to_map,
    _render_report,
    select_workflow_variables,
)
from arc_scope.pipeline.steps import ArcResult
from arc_scope.weather.era5 import _iter_dates, _iter_month_windows


def test_iter_dates_handles_month_boundaries():
    """ERA5 date iteration should include every day across month changes."""
    from datetime import datetime

    dates = _iter_dates(
        datetime(2021, 1, 30, 10, 30),
        datetime(2021, 2, 2, 5, 0),
    )
    assert dates == [
        "2021-01-30",
        "2021-01-31",
        "2021-02-01",
        "2021-02-02",
    ]


def test_select_workflow_variables_prefers_reflectance_matches():
    """Reflectance plotting should prioritise meaningful output names."""
    ds = xr.Dataset(
        {
            "leaf_refl": ("time", np.linspace(0.05, 0.08, 5)),
            "rsot": ("time", np.linspace(0.15, 0.2, 5)),
            "rso": ("time", np.linspace(0.11, 0.16, 5)),
            "temperature": ("time", np.linspace(15.0, 19.0, 5)),
        },
        coords={"time": np.arange(5)},
    )

    selected = select_workflow_variables(ds, "reflectance", limit=3)
    assert selected[:3] == ["rsot", "rso", "leaf_refl"]
    assert len(selected) == 3


def test_iter_month_windows_splits_at_calendar_boundaries():
    """Large ERA5 requests should be broken into month-sized chunks."""
    from datetime import datetime

    windows = _iter_month_windows(
        datetime(2021, 5, 15, 12, 0),
        datetime(2021, 7, 2, 6, 0),
    )
    assert windows == [
        (datetime(2021, 5, 15, 12, 0), datetime(2021, 5, 31, 23, 59, 59, 999999)),
        (datetime(2021, 6, 1, 0, 0), datetime(2021, 6, 30, 23, 59, 59, 999999)),
        (datetime(2021, 7, 1, 0, 0), datetime(2021, 7, 2, 6, 0)),
    ]


def test_render_report_single_workflow_omits_comparison_section():
    """The generated report should describe the real single-workflow path cleanly."""
    time = pd.date_range("2021-06-01", periods=2, freq="D")
    post_bio = xr.DataArray(
        np.ones((1, 1, 3, 2), dtype=np.float64),
        dims=("y", "x", "band", "time"),
        coords={"y": [0], "x": [0], "band": ["lai", "cab", "cw"], "time": time},
    )
    workflow_input = xr.Dataset(
        {
            "LAI": (("time", "y", "x"), np.ones((2, 1, 1), dtype=np.float64)),
            "Cab": (("time", "y", "x"), np.ones((2, 1, 1), dtype=np.float64)),
            "Cw": (("time", "y", "x"), np.ones((2, 1, 1), dtype=np.float64)),
            "Ta": ("time", np.array([18.0, 19.0])),
            "Rin": ("time", np.array([500.0, 520.0])),
            "tts": ("time", np.array([35.0, 37.0])),
        },
        coords={"time": time, "y": [0], "x": [0]},
    )
    workflow_output = xr.Dataset(
        {
            "rsot": (("time", "y", "x", "wavelength"), np.ones((2, 1, 1, 2), dtype=np.float64)),
            "rso": (("time", "y", "x", "wavelength"), np.ones((2, 1, 1, 2), dtype=np.float64)),
        },
        coords={"time": time, "y": [0], "x": [0], "wavelength": [680.0, 740.0]},
    )
    result = DualWorkflowExperimentResult(
        runtime=RuntimeCheck(
            package_versions={"python": "3.11"},
            requirements={"arc": "available", "scope": "available"},
            scope_root="/tmp/SCOPE",
        ),
        config={
            "geojson_path": "field.geojson",
            "location": {"latitude": 51.278, "longitude": 5.019},
            "start_date": "2021-05-15",
            "end_date": "2021-10-01",
            "crop_type": "wheat",
        },
        arc_result=ArcResult(
            scale_data=np.zeros((1, 1), dtype=np.float64),
            post_bio_tensor=np.zeros((1, 1, 1), dtype=np.float64),
            post_bio_unc_tensor=np.zeros((1, 1, 1), dtype=np.float64),
            mask=np.zeros((1, 1), dtype=bool),
            doys=np.array([150, 151]),
        ),
        post_bio_da=post_bio,
        post_bio_scale_da=post_bio.sel(band="lai").drop_vars("band", errors="ignore"),
        weather_ds=xr.Dataset({"Rin": ("time", np.array([500.0, 520.0]))}, coords={"time": time}),
        observation_ds=xr.Dataset(
            {"solar_zenith_angle": ("time", np.array([35.0, 37.0]))},
            coords={"time": time},
        ),
        acquisition_table=pd.DataFrame({"date": ["2021-06-01", "2021-06-02"]}),
        workflow_metrics=pd.DataFrame({"workflow": ["reflectance"]}),
        variable_inventory=pd.DataFrame({"workflow": ["reflectance"]}),
        workflow_runs={
            "reflectance": WorkflowRun(
                scope_input_ds=workflow_input,
                scope_output_ds=workflow_output,
                selected_output_variables=["rsot", "rso"],
            )
        },
    )
    manifest = {
        "run_config": "run_config.json",
        "environment": "environment.json",
        "post_bio": "post_bio.nc",
        "post_bio_scale": "post_bio_scale.nc",
        "weather": "weather.nc",
        "observation": "observation.nc",
        "acquisition_table": "acquisition_table.csv",
        "workflow_metrics": "workflow_metrics.csv",
        "variable_inventory": "variable_inventory.csv",
        "manifest": "artifact_manifest.json",
        "scope_input_reflectance": "scope_input_reflectance.nc",
        "scope_output_reflectance": "scope_output_reflectance.nc",
    }

    report = _render_report(result, manifest)

    assert "ARC retrieves crop biophysical state" in report
    assert "SCOPE reflectance simulation" in report
    assert "Workflow Comparison" not in report


def test_reduce_to_map_uses_nearest_time_match():
    """Snapshot maps should tolerate non-identical ARC and SCOPE timestamps."""
    da = xr.DataArray(
        np.arange(8, dtype=np.float64).reshape(2, 1, 1, 4),
        dims=("time", "y", "x", "wavelength"),
        coords={
            "time": pd.to_datetime(["2021-07-25 10:30", "2021-07-27 10:30"]),
            "y": [0],
            "x": [0],
            "wavelength": [680.0, 700.0, 720.0, 740.0],
        },
    )

    reduced = _reduce_to_map(da, pd.Timestamp("2021-07-26 00:00:00"))

    assert reduced.dims == ("y", "x")
    assert float(reduced.values[0, 0]) == np.mean(np.arange(4, dtype=np.float64))
