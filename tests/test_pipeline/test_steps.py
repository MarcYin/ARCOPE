"""Tests for individual pipeline step functions."""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arc_scope.data import TEST_FIELD_GEOJSON
from arc_scope.pipeline.steps import (
    ArcResult,
    _augment_scope_dataset,
    bridge_arc_to_scope,
    build_observation_dataset,
    fetch_weather,
    prepare_scope_dataset,
    retrieve_arc,
    run_scope_simulation,
)
from arc_scope.pipeline.config import PipelineConfig


# ---------------------------------------------------------------------------
# build_observation_dataset tests
# ---------------------------------------------------------------------------

_DOYS = np.array([150, 160, 170, 180, 190, 200])


def test_build_observation_dataset_shapes():
    """Observation dataset should have the right variables and matching sizes."""
    ds = build_observation_dataset(
        doys=_DOYS, year=2021, geojson_path=TEST_FIELD_GEOJSON,
    )
    assert "solar_zenith_angle" in ds
    assert "viewing_zenith_angle" in ds
    assert "solar_azimuth_angle" in ds
    assert "viewing_azimuth_angle" in ds
    assert "delta_time" in ds
    assert ds.sizes["time"] == len(_DOYS)


def test_build_observation_dataset_solar_angles_reasonable():
    """Solar zenith angles for summer mid-latitudes should be < 90 degrees."""
    ds = build_observation_dataset(
        doys=_DOYS, year=2021, geojson_path=TEST_FIELD_GEOJSON,
    )
    sza = ds["solar_zenith_angle"].values
    # All summer observations in the Netherlands at ~10:30 should be < 70 deg
    assert np.all(sza > 0), "SZA should be positive"
    assert np.all(sza < 90), "SZA should be < 90 during daytime"


def test_build_observation_dataset_time_coords():
    """Time coordinates should be datetime64 values in the correct year."""
    ds = build_observation_dataset(
        doys=_DOYS, year=2021, geojson_path=TEST_FIELD_GEOJSON,
    )
    times = ds.coords["time"].values
    assert len(times) == len(_DOYS)
    for t in times:
        year_str = str(t)[:4]
        assert year_str == "2021"


def test_build_observation_dataset_offsets_duplicate_days():
    """Repeated DOYs should produce unique timestamps for SCOPE alignment."""
    ds = build_observation_dataset(
        doys=np.array([150, 150, 151]),
        year=2021,
        geojson_path=TEST_FIELD_GEOJSON,
    )
    times = ds.indexes["time"]
    assert times.is_unique
    assert times[1] - times[0] == pd.Timedelta(minutes=5)
    assert ds.attrs["overpass_time_reference"] == "local solar time"
    assert ds.attrs["time_coordinate_reference"] == "UTC"


def test_build_observation_dataset_overpass_hour_is_local_solar_time(monkeypatch, tmp_path):
    """overpass_hour should drive solar geometry as local solar time, not UTC."""
    monkeypatch.setattr(
        "arc_scope.pipeline.steps.load_geojson_centroid",
        lambda _: (15.0, 0.0),
    )

    ds = build_observation_dataset(
        doys=np.array([80]),
        year=2021,
        geojson_path=tmp_path / "field.geojson",
        overpass_hour=12.0,
    )

    assert float(ds["solar_zenith_angle"].values[0]) < 1.0
    assert pd.Timestamp(ds.indexes["time"][0]).hour == 11


def test_build_observation_dataset_western_longitude_overpass_is_daytime(
    monkeypatch, tmp_path
):
    """Western longitudes should not reinterpret Sentinel-2 LST as pre-dawn UTC."""
    monkeypatch.setattr(
        "arc_scope.pipeline.steps.load_geojson_centroid",
        lambda _: (-96.0, 41.0),
    )

    ds = build_observation_dataset(
        doys=np.array([150]),
        year=2021,
        geojson_path=tmp_path / "field.geojson",
        overpass_hour=10.5,
    )

    timestamp = pd.Timestamp(ds.indexes["time"][0])
    assert timestamp.hour == 16
    assert timestamp.minute == 51
    assert float(ds["solar_zenith_angle"].values[0]) < 45.0


# ---------------------------------------------------------------------------
# bridge_arc_to_scope tests
# ---------------------------------------------------------------------------


def test_bridge_arc_to_scope_with_fixtures(sample_arc_outputs):
    """bridge_arc_to_scope should produce well-shaped DataArrays from an ArcResult."""
    out = sample_arc_outputs
    arc_result = ArcResult(
        scale_data=out["scale_data"],
        post_bio_tensor=out["post_bio_tensor"],
        post_bio_unc_tensor=out["post_bio_unc_tensor"],
        mask=out["mask"],
        doys=out["doys"],
        geotransform=out["geotransform"],
        crs=out["crs"],
    )
    bio_da, scale_da = bridge_arc_to_scope(arc_result, year=2021)
    assert isinstance(bio_da, xr.DataArray)
    assert isinstance(scale_da, xr.DataArray)
    assert bio_da.dims == ("y", "x", "band", "time")
    assert scale_da.dims == ("y", "x", "band")


def test_bridge_arc_to_scope_no_geotransform_raises(sample_arc_outputs):
    """bridge_arc_to_scope should raise when geotransform is None."""
    out = sample_arc_outputs
    arc_result = ArcResult(
        scale_data=out["scale_data"],
        post_bio_tensor=out["post_bio_tensor"],
        post_bio_unc_tensor=out["post_bio_unc_tensor"],
        mask=out["mask"],
        doys=out["doys"],
        geotransform=None,
        crs=out["crs"],
    )
    with pytest.raises(ValueError, match="geotransform"):
        bridge_arc_to_scope(arc_result, year=2021)


# ---------------------------------------------------------------------------
# prepare_scope_dataset tests
# ---------------------------------------------------------------------------


def test_prepare_scope_dataset_broadcasts_time_weather_to_scope_grid(
    monkeypatch, tmp_path
):
    """Field-scale weather should be explicit on the SCOPE y/x/time grid."""
    captured: dict[str, xr.Dataset] = {}

    def fake_prepare_scope_input_dataset(
        *,
        weather_ds,
        observation_ds,
        post_bio_da,
        post_bio_scale_da,
        scope_root_path,
        scope_options,
    ):
        captured["weather_ds"] = weather_ds
        captured["observation_ds"] = observation_ds
        return xr.Dataset({"Rin": weather_ds["Rin"]})

    scope_module = types.ModuleType("scope")
    scope_io_module = types.ModuleType("scope.io")
    scope_prepare_module = types.ModuleType("scope.io.prepare")
    scope_prepare_module.prepare_scope_input_dataset = fake_prepare_scope_input_dataset
    scope_io_module.prepare = scope_prepare_module
    scope_module.io = scope_io_module
    monkeypatch.setitem(sys.modules, "scope", scope_module)
    monkeypatch.setitem(sys.modules, "scope.io", scope_io_module)
    monkeypatch.setitem(sys.modules, "scope.io.prepare", scope_prepare_module)

    times = pd.date_range("2021-06-01", periods=3, freq="D")
    post_bio_da = xr.DataArray(
        np.ones((2, 3, 1, 3), dtype=np.float64),
        dims=("y", "x", "band", "time"),
        coords={"y": [10, 11], "x": [20, 21, 22], "band": ["LAI"], "time": times},
    )
    post_bio_scale_da = xr.DataArray(
        np.ones((2, 3, 1), dtype=np.float64),
        dims=("y", "x", "band"),
        coords={"y": [10, 11], "x": [20, 21, 22], "band": ["SMC"]},
    )
    weather_ds = xr.Dataset(
        {"Rin": ("time", np.array([500.0, 550.0, 600.0]))},
        coords={"time": times},
    )
    observation_ds = xr.Dataset(
        {"tts": ("time", np.array([30.0, 31.0, 32.0]))},
        coords={"time": times},
    )
    config = PipelineConfig(
        geojson_path=str(TEST_FIELD_GEOJSON),
        start_date="2021-06-01",
        end_date="2021-06-03",
        crop_type="wheat",
        start_of_season=120,
        year=2021,
        output_dir=tmp_path,
    )

    prepared = prepare_scope_dataset(
        post_bio_da,
        post_bio_scale_da,
        weather_ds,
        observation_ds,
        config,
    )

    rin = captured["weather_ds"]["Rin"]
    assert rin.dims == ("y", "x", "time")
    assert rin.sizes == {"y": 2, "x": 3, "time": 3}
    np.testing.assert_allclose(rin.sel(y=10, x=20).values, [500.0, 550.0, 600.0])
    tts = captured["observation_ds"]["tts"]
    assert tts.dims == ("y", "x", "time")
    assert tts.sizes == {"y": 2, "x": 3, "time": 3}
    np.testing.assert_allclose(tts.sel(y=11, x=22).values, [30.0, 31.0, 32.0])
    assert prepared["Rin"].dims == ("y", "x", "time")


def test_prepare_scope_dataset_seeds_energy_balance_defaults_after_prepare_retry(
    monkeypatch,
    tmp_path,
):
    """calc_ebal validation in scope-rtm should not block arcope defaults."""
    prepare_options: list[dict[str, object]] = []
    times = pd.date_range("2021-06-01", periods=2, freq="D")

    def fake_prepare_scope_input_dataset(
        *,
        weather_ds,
        observation_ds,
        post_bio_da,
        post_bio_scale_da,
        scope_root_path,
        scope_options,
    ):
        prepare_options.append(dict(scope_options))
        if scope_options.get("calc_ebal"):
            raise ValueError(
                "Dataset validation failed:\n"
                "- Missing required variable Ca: Ambient CO2 concentration.\n"
                "- Missing required variable Cd: Drag coefficient.\n"
                "- Missing required variable d, h, rbs, rss, rwc, z, z0m"
            )
        return xr.Dataset(
            {
                "Rin": (("y", "x", "time"), np.full((1, 2, 2), 625.0)),
                "tts": (("y", "x", "time"), np.full((1, 2, 2), 33.0)),
                "LAI": (("y", "x", "time"), np.full((1, 2, 2), 3.0)),
                "Cab": (("y", "x", "time"), np.full((1, 2, 2), 42.0)),
                "Cw": (("y", "x", "time"), np.full((1, 2, 2), 0.015)),
                "SMC": (("y", "x", "time"), np.full((1, 2, 2), 0.25)),
            },
            coords={"y": [0], "x": [0, 1], "time": times},
        )

    def fake_spectral_forcing(*, rin, sza, time_coord, atmos_file, scope_root_path):
        return xr.Dataset(
            {
                "Esun_sw": rin.expand_dims(wavelength=[680.0]),
                "Esky_sw": rin.expand_dims(wavelength=[680.0]) * 0.2,
                "Esun_": rin.expand_dims(excitation_wavelength=[740.0]),
                "Esky_": rin.expand_dims(excitation_wavelength=[740.0]) * 0.2,
            }
        )

    scope_module = types.ModuleType("scope")
    scope_io_module = types.ModuleType("scope.io")
    scope_prepare_module = types.ModuleType("scope.io.prepare")
    scope_prepare_module.prepare_scope_input_dataset = fake_prepare_scope_input_dataset
    scope_io_module.prepare = scope_prepare_module
    scope_module.io = scope_io_module
    monkeypatch.setitem(sys.modules, "scope", scope_module)
    monkeypatch.setitem(sys.modules, "scope.io", scope_io_module)
    monkeypatch.setitem(sys.modules, "scope.io.prepare", scope_prepare_module)
    monkeypatch.setattr(
        "arc_scope.pipeline.steps.build_scope_spectral_forcing",
        fake_spectral_forcing,
    )

    post_bio_da = xr.DataArray(
        np.ones((1, 2, 1, 2), dtype=np.float64),
        dims=("y", "x", "band", "time"),
        coords={"y": [0], "x": [0, 1], "band": ["LAI"], "time": times},
    )
    post_bio_scale_da = xr.DataArray(
        np.ones((1, 2, 1), dtype=np.float64),
        dims=("y", "x", "band"),
        coords={"y": [0], "x": [0, 1], "band": ["SMC"]},
    )
    config = PipelineConfig(
        geojson_path=str(TEST_FIELD_GEOJSON),
        start_date="2021-06-01",
        end_date="2021-06-02",
        crop_type="soybean",
        start_of_season=120,
        year=2021,
        scope_workflow="fluorescence",
        scope_options={"calc_ebal": 1},
        output_dir=tmp_path,
    )

    prepared = prepare_scope_dataset(
        post_bio_da,
        post_bio_scale_da,
        xr.Dataset({"Rin": ("time", np.full(2, 625.0))}, coords={"time": times}),
        xr.Dataset({"tts": ("time", np.full(2, 33.0))}, coords={"time": times}),
        config,
    )

    assert [options["calc_ebal"] for options in prepare_options] == [1, 0]
    for name in ("Ca", "Oa", "Cd", "rwc", "z0m", "d", "h", "rss", "rbs"):
        assert name in prepared
        assert prepared[name].dims == ("y", "x", "time")
        assert bool(np.isfinite(prepared[name]).all().item())
    assert "fqe" in prepared
    assert "Esun_sw" in prepared


# ---------------------------------------------------------------------------
# fetch_weather tests
# ---------------------------------------------------------------------------


def test_fetch_weather_unknown_provider_raises():
    """An unknown weather_provider name should raise ValueError."""
    config = PipelineConfig(
        geojson_path=str(TEST_FIELD_GEOJSON),
        start_date="2021-05-15",
        end_date="2021-10-01",
        crop_type="wheat",
        start_of_season=170,
        year=2021,
        weather_provider="nonexistent_provider",
    )
    with pytest.raises(ValueError, match="Unknown weather provider"):
        fetch_weather(config)


def test_retrieve_arc_passes_data_source(tmp_path, monkeypatch):
    """retrieve_arc should forward config.data_source to ARC."""
    called: dict[str, object] = {}

    def fake_arc_field(**kwargs):
        called.update(kwargs)
        return (
            np.zeros((1, 15), dtype=np.float64),
            np.zeros((1, 7, 1), dtype=np.float64),
            np.zeros((1, 7, 1), dtype=np.float64),
            np.zeros((1, 1), dtype=bool),
            np.array([170], dtype=int),
        )

    monkeypatch.setitem(sys.modules, "arc", types.SimpleNamespace(arc_field=fake_arc_field))
    config = PipelineConfig(
        geojson_path=str(TEST_FIELD_GEOJSON),
        start_date="2021-05-15",
        end_date="2021-10-01",
        crop_type="wheat",
        start_of_season=170,
        year=2021,
        data_source="planetary",
        output_dir=tmp_path,
    )

    retrieve_arc(config)
    assert called["data_source"] == "planetary"


def test_augment_scope_dataset_adds_fluorescence_inputs(tmp_path):
    """Fluorescence runs should gain spectral forcing and fqe."""
    radiation_dir = tmp_path / "radiationdata"
    radiation_dir.mkdir()
    np.savetxt(radiation_dir / "Esun_.dat", np.linspace(100.0, 20.0, 32))
    np.savetxt(radiation_dir / "Esky_.dat", np.linspace(60.0, 10.0, 32))
    (radiation_dir / "dummy.atm").write_text("atm", encoding="utf-8")

    time = pd.date_range("2021-06-01", periods=2, freq="D")
    base = xr.Dataset(
        {
            "Rin": (("time", "y", "x"), np.full((2, 1, 2), 600.0)),
            "tts": (("time", "y", "x"), np.full((2, 1, 2), 35.0)),
            "LAI": (("time", "y", "x"), np.array([[[2.0, 2.5]], [[3.0, 2.8]]])),
            "Cab": (("time", "y", "x"), np.array([[[35.0, 40.0]], [[42.0, 45.0]]])),
            "Cw": (("time", "y", "x"), np.array([[[0.012, 0.013]], [[0.014, 0.015]]])),
            "Ta": (("time", "y", "x"), np.full((2, 1, 2), 20.0)),
            "u": (("time", "y", "x"), np.full((2, 1, 2), 2.0)),
            "SMC": (("time", "y", "x"), np.full((2, 1, 2), 0.25)),
        },
        coords={"time": time, "y": [0], "x": [0, 1]},
        attrs={"atmos_file": str(radiation_dir / "dummy.atm")},
    )
    config = PipelineConfig(
        geojson_path=str(TEST_FIELD_GEOJSON),
        start_date="2021-05-15",
        end_date="2021-10-01",
        crop_type="wheat",
        start_of_season=170,
        year=2021,
        scope_workflow="fluorescence",
        output_dir=tmp_path,
    )

    augmented = _augment_scope_dataset(base, config)

    for name in ("Esun_sw", "Esky_sw", "Esun_", "Esky_", "fqe"):
        assert name in augmented
        assert bool(np.isfinite(augmented[name]).all().item())
    assert "wavelength" in augmented["Esun_sw"].dims
    assert "excitation_wavelength" in augmented["Esun_"].dims


def test_augment_scope_dataset_adds_thermal_state(tmp_path):
    """Thermal runs should gain diagnostic canopy and soil temperatures."""
    radiation_dir = tmp_path / "radiationdata"
    radiation_dir.mkdir()
    np.savetxt(radiation_dir / "Esun_.dat", np.linspace(100.0, 20.0, 32))
    np.savetxt(radiation_dir / "Esky_.dat", np.linspace(60.0, 10.0, 32))
    (radiation_dir / "dummy.atm").write_text("atm", encoding="utf-8")

    time = pd.date_range("2021-06-01", periods=2, freq="D")
    base = xr.Dataset(
        {
            "Rin": (("time", "y", "x"), np.full((2, 2, 1), 650.0)),
            "tts": (("time", "y", "x"), np.full((2, 2, 1), 30.0)),
            "LAI": (("time", "y", "x"), np.array([[[1.5], [2.2]], [[2.8], [3.1]]])),
            "Cab": (("time", "y", "x"), np.full((2, 2, 1), 40.0)),
            "Cw": (("time", "y", "x"), np.array([[[0.010], [0.011]], [[0.013], [0.014]]])),
            "Ta": (("time", "y", "x"), np.full((2, 2, 1), 18.0)),
            "u": (("time", "y", "x"), np.full((2, 2, 1), 2.5)),
            "SMC": (("time", "y", "x"), np.array([[[0.20], [0.22]], [[0.24], [0.28]]])),
        },
        coords={"time": time, "y": [0, 1], "x": [0]},
        attrs={"atmos_file": str(radiation_dir / "dummy.atm")},
    )
    config = PipelineConfig(
        geojson_path=str(TEST_FIELD_GEOJSON),
        start_date="2021-05-15",
        end_date="2021-10-01",
        crop_type="wheat",
        start_of_season=170,
        year=2021,
        scope_workflow="thermal",
        output_dir=tmp_path,
    )

    augmented = _augment_scope_dataset(base, config)

    for name in ("Tcu", "Tch", "Tsu", "Tsh"):
        assert name in augmented
        assert bool(np.isfinite(augmented[name]).all().item())
    assert bool((augmented["Tcu"] >= augmented["Tch"]).all().item())
    assert bool((augmented["Tsu"] >= augmented["Tsh"]).all().item())


def test_augment_scope_dataset_adds_energy_balance_state(tmp_path):
    """Coupled energy balance should derive its extra aero/biochem inputs."""
    radiation_dir = tmp_path / "radiationdata"
    radiation_dir.mkdir()
    np.savetxt(radiation_dir / "Esun_.dat", np.linspace(100.0, 20.0, 32))
    np.savetxt(radiation_dir / "Esky_.dat", np.linspace(60.0, 10.0, 32))
    (radiation_dir / "dummy.atm").write_text("atm", encoding="utf-8")

    time = pd.date_range("2021-06-01", periods=2, freq="D")
    base = xr.Dataset(
        {
            "Rin": (("time", "y", "x"), np.full((2, 1, 2), 625.0)),
            "tts": (("time", "y", "x"), np.full((2, 1, 2), 33.0)),
            "LAI": (("time", "y", "x"), np.array([[[2.0, 2.6]], [[3.1, 3.7]]])),
            "Cab": (("time", "y", "x"), np.full((2, 1, 2), 42.0)),
            "Cw": (("time", "y", "x"), np.full((2, 1, 2), 0.015)),
            "Cdm": (("time", "y", "x"), np.full((2, 1, 2), 0.004)),
            "Ta": (("time", "y", "x"), np.full((2, 1, 2), 20.0)),
            "ea": (("time", "y", "x"), np.full((2, 1, 2), 14.0)),
            "p": (("time", "y", "x"), np.full((2, 1, 2), 1011.0)),
            "u": (("time", "y", "x"), np.full((2, 1, 2), 2.3)),
            "tto": (("time", "y", "x"), np.zeros((2, 1, 2), dtype=np.float64)),
            "psi": (("time", "y", "x"), np.full((2, 1, 2), 210.0)),
            "SMC": (("time", "y", "x"), np.array([[[0.18, 0.22]], [[0.26, 0.31]]])),
        },
        coords={"time": time, "y": [0], "x": [0, 1]},
        attrs={"atmos_file": str(radiation_dir / "dummy.atm")},
    )
    config = PipelineConfig(
        geojson_path=str(TEST_FIELD_GEOJSON),
        start_date="2021-05-15",
        end_date="2021-10-01",
        crop_type="wheat",
        start_of_season=170,
        year=2021,
        scope_workflow="energy-balance",
        output_dir=tmp_path,
    )

    augmented = _augment_scope_dataset(base, config)

    for name in (
        "Esun_sw",
        "Esky_sw",
        "Esun_",
        "Esky_",
        "fqe",
        "Ca",
        "Oa",
        "z",
        "Cd",
        "rwc",
        "z0m",
        "d",
        "h",
        "rss",
        "rbs",
        "Vcmax25",
        "BallBerrySlope",
    ):
        assert name in augmented
        assert bool(np.isfinite(augmented[name]).all().item())

    for name in ("Tcu", "Tch", "Tsu", "Tsh"):
        assert name not in augmented


def test_run_scope_simulation_routes_energy_balance_to_coupled_runner(monkeypatch):
    """Energy-balance config should bypass the generic scope dispatch."""
    calls: dict[str, object] = {"validations": []}

    class FakeSimulationConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeScopeGridDataModule:
        def __init__(self, dataset, sim_config, required_vars):
            self.dataset = dataset
            self.sim_config = sim_config
            self.required_vars = required_vars
            calls["sim_config"] = sim_config

    class FakeRunner:
        def __init__(self):
            self.calls: list[tuple[str, object]] = []

        @classmethod
        def from_scope_assets(cls, **kwargs):
            instance = cls()
            instance.init_kwargs = kwargs
            calls["runner"] = instance
            return instance

        def run_scope_dataset(self, *args, **kwargs):
            raise AssertionError("run_scope_dataset should not be used for energy-balance")

        def run_energy_balance_fluorescence_dataset(self, *args, **kwargs):
            self.calls.append(("fluorescence", kwargs["soil_heat_method"]))
            return xr.Dataset(
                {"F685": (("time", "y", "x"), np.ones((2, 1, 1), dtype=np.float64))},
                coords={"time": pd.date_range("2021-06-01", periods=2, freq="D"), "y": [0], "x": [0]},
                attrs={"scope_product": "energy_balance_fluorescence"},
            )

        def run_energy_balance_thermal_dataset(self, *args, **kwargs):
            self.calls.append(("thermal", kwargs["soil_heat_method"]))
            return xr.Dataset(
                {"Loutt": (("time", "y", "x"), np.full((2, 1, 1), 300.0, dtype=np.float64))},
                coords={"time": pd.date_range("2021-06-01", periods=2, freq="D"), "y": [0], "x": [0]},
                attrs={"scope_product": "energy_balance_thermal"},
            )

    def fake_validate_scope_dataset(dataset, *, workflow, scope_options=None):
        calls["validations"].append((workflow, scope_options))

    fake_scope_module = types.SimpleNamespace(
        SimulationConfig=FakeSimulationConfig,
        ScopeGridRunner=FakeRunner,
        campbell_lidf=lambda angle, device, dtype: ("lidf", angle, device, dtype),
    )
    fake_scope_data = types.SimpleNamespace(ScopeGridDataModule=FakeScopeGridDataModule)
    fake_scope_io = types.SimpleNamespace(validate_scope_dataset=fake_validate_scope_dataset)
    fake_fluspect_module = types.SimpleNamespace(
        FluspectModel=type(
            "FakeFluspectModel",
            (),
            {"_stacked_layers": staticmethod(lambda r, t, N: (r, t))},
        )
    )
    fake_torch_module = types.SimpleNamespace(
        float32="float32",
        float64="float64",
        device=lambda name: name,
    )

    monkeypatch.setitem(sys.modules, "scope", fake_scope_module)
    monkeypatch.setitem(sys.modules, "scope.data", fake_scope_data)
    monkeypatch.setitem(sys.modules, "scope.io", fake_scope_io)
    monkeypatch.setitem(sys.modules, "scope.spectral.fluspect", fake_fluspect_module)
    monkeypatch.setitem(sys.modules, "torch", fake_torch_module)

    scope_dataset = xr.Dataset(
        {
            "Cab": (("time", "y", "x"), np.full((2, 1, 1), 40.0)),
            "Cw": (("time", "y", "x"), np.full((2, 1, 1), 0.015)),
            "Cdm": (("time", "y", "x"), np.full((2, 1, 1), 0.004)),
            "LAI": (("time", "y", "x"), np.full((2, 1, 1), 3.0)),
            "tts": (("time", "y", "x"), np.full((2, 1, 1), 35.0)),
            "tto": (("time", "y", "x"), np.zeros((2, 1, 1), dtype=np.float64)),
            "psi": (("time", "y", "x"), np.full((2, 1, 1), 210.0)),
            "Ta": (("time", "y", "x"), np.full((2, 1, 1), 19.0)),
            "ea": (("time", "y", "x"), np.full((2, 1, 1), 14.0)),
            "Ca": (("time", "y", "x"), np.full((2, 1, 1), 410.0)),
            "Oa": (("time", "y", "x"), np.full((2, 1, 1), 209.0)),
            "p": (("time", "y", "x"), np.full((2, 1, 1), 1013.0)),
            "z": (("time", "y", "x"), np.full((2, 1, 1), 10.0)),
            "u": (("time", "y", "x"), np.full((2, 1, 1), 2.0)),
            "Cd": (("time", "y", "x"), np.full((2, 1, 1), 0.3)),
            "rwc": (("time", "y", "x"), np.zeros((2, 1, 1), dtype=np.float64)),
            "z0m": (("time", "y", "x"), np.full((2, 1, 1), 0.12)),
            "d": (("time", "y", "x"), np.full((2, 1, 1), 0.65)),
            "h": (("time", "y", "x"), np.full((2, 1, 1), 0.95)),
            "rss": (("time", "y", "x"), np.full((2, 1, 1), 180.0)),
            "rbs": (("time", "y", "x"), np.full((2, 1, 1), 10.0)),
            "Esun_sw": (("time", "y", "x", "wavelength"), np.ones((2, 1, 1, 3), dtype=np.float64)),
            "Esky_sw": (("time", "y", "x", "wavelength"), np.ones((2, 1, 1, 3), dtype=np.float64)),
            "Esun_": (("time", "y", "x", "excitation_wavelength"), np.ones((2, 1, 1, 2), dtype=np.float64)),
            "Esky_": (("time", "y", "x", "excitation_wavelength"), np.ones((2, 1, 1, 2), dtype=np.float64)),
            "fqe": (("time", "y", "x"), np.full((2, 1, 1), 0.01)),
            "Vcmax25": (("time", "y", "x"), np.full((2, 1, 1), 60.0)),
            "BallBerrySlope": (("time", "y", "x"), np.full((2, 1, 1), 8.0)),
            "BSMBrightness": (("time", "y", "x"), np.full((2, 1, 1), 0.5)),
            "BSMlat": (("time", "y", "x"), np.full((2, 1, 1), 15.0)),
            "BSMlon": (("time", "y", "x"), np.full((2, 1, 1), 45.0)),
            "SMC": (("time", "y", "x"), np.full((2, 1, 1), 0.25)),
        },
        coords={
            "time": pd.date_range("2021-06-01", periods=2, freq="D"),
            "y": [0.0],
            "x": [1.0],
            "wavelength": [500.0, 680.0, 800.0],
            "excitation_wavelength": [650.0, 760.0],
        },
    )
    config = PipelineConfig(
        geojson_path=str(TEST_FIELD_GEOJSON),
        start_date="2021-05-15",
        end_date="2021-10-01",
        crop_type="wheat",
        start_of_season=170,
        year=2021,
        scope_workflow="energy-balance",
        scope_chunk_size=64,
    )

    output = run_scope_simulation(scope_dataset, config)

    assert {"F685", "Loutt"}.issubset(output.data_vars)
    assert output.attrs["scope_product"] == "energy_balance"
    assert output.attrs["scope_components"] == "energy,physiology,fluorescence,thermal"
    assert calls["sim_config"].kwargs["chunk_size"] == 64
    assert calls["runner"].calls == [("fluorescence", 2), ("thermal", 2)]
    assert calls["validations"] == [
        ("energy-balance-fluorescence", None),
        ("energy-balance-thermal", None),
    ]
