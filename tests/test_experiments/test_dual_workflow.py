"""Tests for the heavyweight ARC-SCOPE experiment helpers."""

from __future__ import annotations

import json
import pandas as pd
import numpy as np
import xarray as xr

from arc_scope.experiments.dual_workflow import (
    DualWorkflowExperimentResult,
    RuntimeCheck,
    WorkflowRun,
    _build_explorer_payload,
    _reduce_to_map,
    _render_explorer_html,
    _resolve_simulation_subset,
    _render_report,
    _subset_scope_dataset,
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


def test_select_workflow_variables_prefers_fluorescence_and_thermal_signals():
    """Summary plots should surface SIF and thermal outputs before leaf optics."""
    time = np.arange(5)
    fluorescence_ds = xr.Dataset(
        {
            "leaf_fluor_back": ("time", np.linspace(0.01, 0.02, 5)),
            "LoF_": (("time", "fluorescence_wavelength"), np.ones((5, 2), dtype=np.float64)),
            "F685": ("time", np.linspace(0.2, 0.3, 5)),
            "F740": ("time", np.linspace(0.4, 0.5, 5)),
        },
        coords={"time": time, "fluorescence_wavelength": [685.0, 740.0]},
    )
    thermal_ds = xr.Dataset(
        {
            "leaf_refl": ("time", np.linspace(0.05, 0.08, 5)),
            "Lot_": (("time", "thermal_wavelength"), np.ones((5, 2), dtype=np.float64)),
            "Loutt": ("time", np.linspace(280.0, 285.0, 5)),
            "Eoutt": ("time", np.linspace(430.0, 450.0, 5)),
        },
        coords={"time": time, "thermal_wavelength": [8.0, 10.0]},
    )

    fluorescence_selected = select_workflow_variables(fluorescence_ds, "fluorescence", limit=3)
    thermal_selected = select_workflow_variables(thermal_ds, "thermal", limit=3)

    assert fluorescence_selected[:3] == ["LoF_", "F685", "F740"]
    assert thermal_selected[:3] == ["Lot_", "Loutt", "Eoutt"]


def test_select_workflow_variables_prefers_energy_balance_signals():
    """Energy-balance plots should surface coupled SIF, thermal, and flux fields."""
    time = np.arange(5)
    ds = xr.Dataset(
        {
            "LoF_": (("time", "fluorescence_wavelength"), np.ones((5, 2), dtype=np.float64)),
            "Lot_": (("time", "thermal_wavelength"), np.ones((5, 2), dtype=np.float64)),
            "LE": ("time", np.linspace(40.0, 80.0, 5)),
            "H": ("time", np.linspace(10.0, 30.0, 5)),
        },
        coords={
            "time": time,
            "fluorescence_wavelength": [685.0, 740.0],
            "thermal_wavelength": [8.0, 10.0],
        },
    )

    selected = select_workflow_variables(ds, "energy-balance", limit=4)
    assert selected == ["LoF_", "Lot_", "LE", "H"]


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
        "explorer_payload": "explorer_payload.json",
        "explorer": "explorer.html",
        "scope_input_reflectance": "scope_input_reflectance.nc",
        "scope_output_reflectance": "scope_output_reflectance.nc",
    }

    report = _render_report(result, manifest)

    assert "ARC retrieves crop biophysical state" in report
    assert "SCOPE simulates output" in report
    assert "Interactive Explorer" in report
    assert "Workflow Comparison" not in report


def test_simulation_subset_targets_dense_valid_window():
    """The compact SCOPE window should choose the densest valid y/x block."""
    time = pd.date_range("2021-06-01", periods=2, freq="D")
    values = np.full((4, 4, 1, 2), np.nan, dtype=np.float64)
    values[1:3, 1:3, 0, :] = 1.0
    post_bio = xr.DataArray(
        values,
        dims=("y", "x", "band", "time"),
        coords={"y": np.arange(4), "x": np.arange(4), "band": ["lai"], "time": time},
    )

    subset = _resolve_simulation_subset(post_bio_da=post_bio, subset_size=2)
    scope_input = xr.Dataset(
        {
            "LAI": (("time", "y", "x"), np.arange(32, dtype=np.float64).reshape(2, 4, 4)),
        },
        coords={"time": time, "y": np.arange(4), "x": np.arange(4)},
    )
    compact = _subset_scope_dataset(scope_input, subset)

    assert subset["applied"] is True
    assert subset["y_start"] == 1
    assert subset["y_stop"] == 3
    assert subset["x_start"] == 1
    assert subset["x_stop"] == 3
    assert compact.sizes["y"] == 2
    assert compact.sizes["x"] == 2


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


def test_build_explorer_payload_includes_spectral_and_scalar_entries():
    """Explorer payload should serialise browseable y/x/time variables."""
    time = pd.date_range("2021-06-01", periods=3, freq="D")
    wavelengths = [680.0, 700.0, 720.0]
    fluorescence_wavelength = [685.0, 740.0]
    thermal_wavelength = [8.0, 10.0]
    scope_input = xr.Dataset(
        {
            "LAI": (("time", "y", "x"), np.ones((3, 2, 2), dtype=np.float64)),
            "Cab": (("time", "y", "x"), np.full((3, 2, 2), 40.0, dtype=np.float64)),
            "Cw": (("time", "y", "x"), np.full((3, 2, 2), 0.014, dtype=np.float64)),
        },
        coords={"time": time, "y": [0, 1], "x": [0, 1]},
    )
    fluorescence_input = scope_input.assign(
        fqe=(("time", "y", "x"), np.full((3, 2, 2), 0.01, dtype=np.float64))
    )
    thermal_input = scope_input.assign(
        Tcu=(("time", "y", "x"), np.full((3, 2, 2), 24.0, dtype=np.float64)),
        Tch=(("time", "y", "x"), np.full((3, 2, 2), 22.0, dtype=np.float64)),
        Tsu=(("time", "y", "x"), np.full((3, 2, 2), 28.0, dtype=np.float64)),
        Tsh=(("time", "y", "x"), np.full((3, 2, 2), 25.0, dtype=np.float64)),
    )
    reflectance_output = xr.Dataset(
        {
            "rsot": (("time", "y", "x", "wavelength"), np.ones((3, 2, 2, 3), dtype=np.float64)),
        },
        coords={"time": time, "y": [0, 1], "x": [0, 1], "wavelength": wavelengths},
    )
    fluorescence_output = xr.Dataset(
        {
            "LoF_": (
                ("time", "y", "x", "fluorescence_wavelength"),
                np.ones((3, 2, 2, 2), dtype=np.float64),
            ),
            "F685": (("time", "y", "x"), np.full((3, 2, 2), 0.4, dtype=np.float64)),
        },
        coords={
            "time": time,
            "y": [0, 1],
            "x": [0, 1],
            "fluorescence_wavelength": fluorescence_wavelength,
        },
    )
    thermal_output = xr.Dataset(
        {
            "Lot_": (
                ("time", "y", "x", "thermal_wavelength"),
                np.ones((3, 2, 2, 2), dtype=np.float64),
            ),
            "Loutt": (("time", "y", "x"), np.full((3, 2, 2), 280.0, dtype=np.float64)),
        },
        coords={
            "time": time,
            "y": [0, 1],
            "x": [0, 1],
            "thermal_wavelength": thermal_wavelength,
        },
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
            doys=np.array([150, 151, 152]),
        ),
        post_bio_da=xr.DataArray(
            np.ones((2, 2, 3, 3), dtype=np.float64),
            dims=("y", "x", "band", "time"),
            coords={"y": [0, 1], "x": [0, 1], "band": ["lai", "cab", "cw"], "time": time},
        ),
        post_bio_scale_da=xr.DataArray(
            np.ones((2, 2, 1), dtype=np.float64),
            dims=("y", "x", "band"),
            coords={"y": [0, 1], "x": [0, 1], "band": ["BSMBrightness"]},
        ),
        weather_ds=xr.Dataset({"Rin": ("time", np.array([500.0, 520.0, 540.0]))}, coords={"time": time}),
        observation_ds=xr.Dataset(
            {"solar_zenith_angle": ("time", np.array([35.0, 37.0, 39.0]))},
            coords={"time": time},
        ),
        acquisition_table=pd.DataFrame({"date": ["2021-06-01", "2021-06-02", "2021-06-03"]}),
        workflow_metrics=pd.DataFrame({"workflow": ["reflectance", "fluorescence", "thermal"]}),
        variable_inventory=pd.DataFrame({"workflow": ["reflectance", "fluorescence", "thermal"]}),
        workflow_runs={
            "reflectance": WorkflowRun(
                scope_input_ds=scope_input,
                scope_output_ds=reflectance_output,
                selected_output_variables=["rsot"],
            ),
            "fluorescence": WorkflowRun(
                scope_input_ds=fluorescence_input,
                scope_output_ds=fluorescence_output,
                selected_output_variables=["LoF_", "F685"],
            ),
            "thermal": WorkflowRun(
                scope_input_ds=thermal_input,
                scope_output_ds=thermal_output,
                selected_output_variables=["Lot_", "Loutt"],
            ),
        },
    )

    payload = _build_explorer_payload(result)

    assert payload["default_key"] == "reflectance:rsot"
    assert "input:LAI" in payload["variables"]
    assert "fluorescence:LoF_" in payload["variables"]
    assert "thermal:Lot_" in payload["variables"]
    assert payload["variables"]["reflectance:rsot"]["axis_name"] == "wavelength"
    assert payload["variables"]["fluorescence:LoF_"]["axis_name"] == "fluorescence_wavelength"


def test_render_explorer_html_supports_deep_linked_keys():
    """The standalone explorer should accept query-string overrides."""
    html = _render_explorer_html(payload_name="explorer_payload.json")

    assert 'new URLSearchParams(window.location.search)' in html
    assert 'const requestedKey = params.get("key");' in html
    assert 'const requestedGroup = params.get("group");' in html


def test_build_explorer_payload_uses_json_null_for_nan_cells():
    """Explorer payloads must stay valid JSON even when maps contain masked cells."""
    time = pd.date_range("2021-06-01", periods=2, freq="D")
    scope_input = xr.Dataset(
        {
            "LAI": (
                ("time", "y", "x"),
                np.array(
                    [
                        [[1.0, np.nan], [2.0, 3.0]],
                        [[1.5, np.nan], [2.5, 3.5]],
                    ],
                    dtype=np.float64,
                ),
            ),
            "Cab": (("time", "y", "x"), np.full((2, 2, 2), 40.0, dtype=np.float64)),
            "Cw": (("time", "y", "x"), np.full((2, 2, 2), 0.014, dtype=np.float64)),
        },
        coords={"time": time, "y": [0, 1], "x": [0, 1]},
    )
    reflectance_output = xr.Dataset(
        {
            "rsot": (
                ("time", "y", "x", "wavelength"),
                np.array(
                    [
                        [[[0.1, 0.2], [np.nan, np.nan]], [[0.3, 0.4], [0.5, 0.6]]],
                        [[[0.15, 0.25], [np.nan, np.nan]], [[0.35, 0.45], [0.55, 0.65]]],
                    ],
                    dtype=np.float64,
                ),
            ),
        },
        coords={"time": time, "y": [0, 1], "x": [0, 1], "wavelength": [680.0, 740.0]},
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
            "start_date": "2021-06-01",
            "end_date": "2021-06-02",
            "crop_type": "wheat",
        },
        arc_result=ArcResult(
            scale_data=np.zeros((1, 1), dtype=np.float64),
            post_bio_tensor=np.zeros((1, 1, 1), dtype=np.float64),
            post_bio_unc_tensor=np.zeros((1, 1, 1), dtype=np.float64),
            mask=np.zeros((1, 1), dtype=bool),
            doys=np.array([152, 153]),
        ),
        post_bio_da=xr.DataArray(
            np.ones((2, 2, 3, 2), dtype=np.float64),
            dims=("y", "x", "band", "time"),
            coords={"y": [0, 1], "x": [0, 1], "band": ["lai", "cab", "cw"], "time": time},
        ),
        post_bio_scale_da=xr.DataArray(
            np.ones((2, 2, 1), dtype=np.float64),
            dims=("y", "x", "band"),
            coords={"y": [0, 1], "x": [0, 1], "band": ["BSMBrightness"]},
        ),
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
                scope_input_ds=scope_input,
                scope_output_ds=reflectance_output,
                selected_output_variables=["rsot"],
            )
        },
    )

    payload = _build_explorer_payload(result)
    encoded = json.dumps(payload, allow_nan=False)

    assert "NaN" not in encoded
    assert "null" in encoded
