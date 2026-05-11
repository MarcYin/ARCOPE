"""Tests for the tensor-preserving SCOPE optimisation path."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from arc_scope.optim.objective import ScopeObjective
from arc_scope.optim.parameters import ParameterSet, ParameterSpec
from arc_scope.pipeline.config import PipelineConfig
from arc_scope.pipeline.optimization import (
    _build_optimizer,
    _default_torch_scope_runner,
    _resolve_autograd_jac_setting,
)
from arc_scope.pipeline.steps import (
    _TensorScopeGridDataModule,
    _resolve_scope_chunk_size,
    _seed_missing_parameter_variables,
    _to_torch_tensor_preserving_graph,
)


class FakeTensor:
    def __init__(self, values, *, requires_grad: bool = False):
        self.values = np.asarray(values, dtype=np.float64)
        self.requires_grad = requires_grad

    def to(self, *, dtype=None, device=None):
        return self

    def isfinite(self):
        return FakeTensor(np.isfinite(self.values), requires_grad=False)

    def all(self):
        return bool(np.all(self.values))


class FakeTorch:
    Tensor = FakeTensor

    @staticmethod
    def as_tensor(values, *, dtype=None, device=None):
        if isinstance(values, FakeTensor):
            return values.to(dtype=dtype, device=device)
        return FakeTensor(values)

    @staticmethod
    def nan_to_num(tensor, *, nan=0.0):
        return FakeTensor(
            np.nan_to_num(tensor.values, nan=nan),
            requires_grad=tensor.requires_grad,
        )


class FakeDataArray:
    def __init__(self, data):
        self.data = data

    @property
    def values(self):
        raise AssertionError(".values would detach a graph-backed tensor")


def test_to_torch_tensor_preserving_graph_uses_data_not_values():
    tensor = FakeTensor([1.0, 2.0], requires_grad=True)
    config = SimpleNamespace(dtype="float64", torch_device=lambda: "cpu")

    result = _to_torch_tensor_preserving_graph(
        FakeDataArray(tensor),
        torch_module=FakeTorch,
        config=config,
    )

    assert result is tensor
    assert result.requires_grad is True


def test_tensor_data_module_overlays_parameters_without_xarray_tensor_storage():
    torch = pytest.importorskip("torch")
    dataset = xr.Dataset(
        {
            "fqe": (("time", "y", "x"), np.full((2, 1, 2), 0.01)),
            "LAI": (("time", "y", "x"), np.full((2, 1, 2), 3.0)),
        },
        coords={"time": [0, 1], "y": [0], "x": [0, 1]},
    )

    class Config(SimpleNamespace):
        def chunks(self, total):
            return [slice(0, total)]

    parameter = torch.tensor(0.02, dtype=torch.float64, requires_grad=True)
    data_module = _TensorScopeGridDataModule(
        dataset,
        Config(dtype=torch.float64, torch_device=lambda: "cpu"),
        required_vars=["fqe", "LAI"],
        torch_module=torch,
        parameter_values={"fqe": parameter},
    )

    batch = next(data_module.iter_batches())

    assert isinstance(dataset["fqe"].data, np.ndarray)
    assert batch["fqe"].requires_grad is True
    assert tuple(batch["fqe"].shape) == (4,)
    batch["fqe"].sum().backward()
    assert float(parameter.grad) == pytest.approx(4.0)


def test_missing_tensor_runner_parameter_placeholder_uses_scope_grid():
    dataset = xr.Dataset(
        {
            "LAI": (("y", "x", "time"), np.full((2, 1, 3), 3.0)),
        },
        coords={"y": [10, 11], "x": [20], "time": [0, 1, 2]},
    )

    seeded = _seed_missing_parameter_variables(
        dataset,
        {"Vcmax25": 40.0},
    )

    assert seeded["Vcmax25"].dims == ("y", "x", "time")
    assert seeded["Vcmax25"].sizes == {"y": 2, "x": 1, "time": 3}


def test_missing_tensor_runner_parameter_is_broadcast_in_tensor_batches():
    torch = pytest.importorskip("torch")
    dataset = xr.Dataset(
        {
            "LAI": (("y", "x", "time"), np.full((2, 1, 3), 3.0)),
        },
        coords={"y": [10, 11], "x": [20], "time": [0, 1, 2]},
    )

    parameter = torch.tensor(40.0, dtype=torch.float64, requires_grad=True)
    seeded = _seed_missing_parameter_variables(
        dataset,
        {"Vcmax25": parameter},
    )

    class Config(SimpleNamespace):
        def chunks(self, total):
            return [slice(0, total)]

    data_module = _TensorScopeGridDataModule(
        seeded,
        Config(dtype=torch.float64, torch_device=lambda: "cpu"),
        required_vars=["LAI", "Vcmax25"],
        torch_module=torch,
        parameter_values={"Vcmax25": parameter},
    )

    batch = next(data_module.iter_batches())

    assert tuple(batch["Vcmax25"].shape) == (6,)
    assert batch["Vcmax25"].requires_grad is True
    assert batch["Vcmax25"].detach().cpu().numpy().tolist() == pytest.approx([40.0] * 6)
    batch["Vcmax25"].sum().backward()
    assert float(parameter.grad) == pytest.approx(6.0)


def test_scope_objective_torch_runner_gets_params_without_dataset_injection():
    torch = pytest.importorskip("torch")
    base_ds = xr.Dataset({"template": ("time", np.ones(3))})
    obs_ds = xr.Dataset({"target": ("time", np.full(3, 4.0))})
    params = ParameterSet(
        [ParameterSpec("x", initial=1.0, lower=-10.0, upper=10.0)]
    )

    def torch_runner(dataset, torch_params):
        assert "x" not in dataset
        return {
            "target": torch_params["x"]
            * torch.ones(3, dtype=torch.float64),
        }

    objective = ScopeObjective(
        base_dataset=base_ds,
        observations=obs_ds,
        target_variables=["target"],
        scope_runner=lambda dataset: xr.Dataset(
            {"target": ("time", np.ones(3))},
            coords={"time": [0, 1, 2]},
        ),
        torch_scope_runner=torch_runner,
    )

    value, gradient = objective.evaluate_value_and_gradient(params.to_array(), params)

    assert value == pytest.approx(9.0)
    assert gradient.tolist() == pytest.approx([-6.0])


def test_scope_objective_streams_chunk_losses_before_returning_gradient(monkeypatch):
    torch = pytest.importorskip("torch")
    base_ds = xr.Dataset({"template": ("sample", np.ones(5))})
    obs_ds = xr.Dataset({"target": ("sample", np.zeros(5))})
    params = ParameterSet(
        [ParameterSpec("scale", initial=2.0, lower=0.1, upper=10.0, transform="log")]
    )
    events = []

    class StreamingRunner:
        def __init__(self):
            self.iter_calls = 0
            self.chunk_sizes = []

        def __call__(self, dataset, torch_params):
            scale = torch_params["scale"]
            values = torch.arange(1, 6, dtype=torch.float64)
            return {"target": scale * values}

        def iter_chunks(self, dataset, torch_params):
            self.iter_calls += 1
            scale = torch_params["scale"]
            values = torch.arange(1, 6, dtype=torch.float64)
            for start, stop in ((0, 2), (2, 4), (4, 5)):
                self.chunk_sizes.append(stop - start)
                events.append(f"yield:{start}:{stop}")
                yield {"target": scale * values[start:stop]}

    original_grad = torch.autograd.grad
    retain_graph_values = []

    def checked_grad(*args, **kwargs):
        retain_graph_values.append(kwargs.get("retain_graph"))
        events.append("grad")
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", checked_grad)

    runner = StreamingRunner()
    objective = ScopeObjective(
        base_dataset=base_ds,
        observations=obs_ds,
        target_variables=["target"],
        scope_runner=lambda dataset: xr.Dataset(
            {"target": ("sample", np.ones(5))},
            coords={"sample": np.arange(5)},
        ),
        torch_scope_runner=runner,
    )

    value, gradient = objective.evaluate_value_and_gradient(params.to_array(), params)

    assert value == pytest.approx(44.0)
    assert gradient.tolist() == pytest.approx([88.0])
    assert runner.iter_calls == 2
    assert runner.chunk_sizes == [2, 2, 1, 2, 2, 1]
    assert len(retain_graph_values) == 3
    assert retain_graph_values == [True, True, True]
    assert events == [
        "yield:0:2",
        "yield:2:4",
        "yield:4:5",
        "yield:0:2",
        "grad",
        "yield:2:4",
        "grad",
        "yield:4:5",
        "grad",
    ]


def test_scope_objective_streaming_spatial_selector_matches_full_torch_loss():
    torch = pytest.importorskip("torch")
    values = torch.arange(12, dtype=torch.float64).reshape(2, 2, 3)
    base_ds = xr.Dataset(
        {"template": (("y", "x", "time"), np.ones((2, 2, 3)))},
        coords={"y": [10, 11], "x": [20, 21], "time": [0, 1, 2]},
    )
    obs_ds = xr.Dataset(
        {"target": ("time", np.array([7.0, 9.0, 11.0]))},
        coords={"time": [0, 1, 2]},
    )
    params = ParameterSet(
        [ParameterSpec("scale", initial=1.0, lower=-10.0, upper=10.0)]
    )

    class StreamingRunner:
        def __call__(self, dataset, torch_params):
            scale = torch_params["scale"]
            return {"target": (scale * values).reshape(-1)}

        def iter_chunks(self, dataset, torch_params):
            scale = torch_params["scale"]
            flat = (scale * values).reshape(-1)
            for start, stop in ((0, 4), (4, 8), (8, 12)):
                yield {"target": flat[start:stop]}

    objective = ScopeObjective(
        base_dataset=base_ds,
        observations=obs_ds,
        target_variables=["target"],
        torch_scope_runner=StreamingRunner(),
        pixel_selector={"y": 11, "x": 20},
    )
    full_tensor = params.to_torch()
    full_loss = objective.evaluate_torch({}, full_tensor, params)
    full_loss.backward()

    stream_value, stream_gradient = objective.evaluate_value_and_gradient(
        params.to_array(),
        params,
    )

    assert stream_value == pytest.approx(float(full_loss.detach().cpu().item()))
    assert stream_gradient.tolist() == pytest.approx(
        full_tensor.grad.detach().cpu().numpy().tolist()
    )


def test_default_torch_scope_runner_exposes_chunk_iterator(tmp_path):
    config = PipelineConfig(
        geojson_path=tmp_path / "field.geojson",
        start_date="2021-06-01",
        end_date="2021-06-02",
        crop_type="wheat",
        start_of_season=120,
        year=2021,
    )

    runner = _default_torch_scope_runner(config)

    assert callable(runner)
    assert callable(getattr(runner, "iter_chunks", None))


def test_real_scope_pipeline_defaults_to_required_autograd_jac():
    optimizer, _ = _build_optimizer({}, default_autograd_jac="required")

    assert optimizer._autograd_required is True
    assert _resolve_autograd_jac_setting({}, default="required") == "required"


def test_proxy_pipeline_can_keep_auto_autograd_jac_default():
    optimizer, _ = _build_optimizer({}, default_autograd_jac="auto")

    assert optimizer._autograd_required is False
    assert optimizer._autograd_disabled is False
    assert _resolve_autograd_jac_setting({}, default="auto") == "auto"


def test_explicit_autograd_jac_setting_overrides_pipeline_default():
    optim_config = {"optimizer": {"type": "scipy", "use_autograd_jac": False}}
    optimizer, _ = _build_optimizer(optim_config, default_autograd_jac="required")

    assert optimizer._autograd_disabled is True
    assert _resolve_autograd_jac_setting(optim_config, default="required") is False


def test_scope_chunk_size_defaults_to_pipeline_setting(tmp_path):
    config = PipelineConfig(
        geojson_path=tmp_path / "field.geojson",
        start_date="2021-06-01",
        end_date="2021-06-02",
        crop_type="wheat",
        start_of_season=120,
        year=2021,
        scope_chunk_size=256,
    )

    assert _resolve_scope_chunk_size(config) == 256


def test_optimization_chunk_size_overrides_scope_setting(tmp_path):
    config = PipelineConfig(
        geojson_path=tmp_path / "field.geojson",
        start_date="2021-06-01",
        end_date="2021-06-02",
        crop_type="wheat",
        start_of_season=120,
        year=2021,
        scope_chunk_size=1024,
        optim_config={"optim": {"enabled": True, "batch_size": "128"}},
    )

    assert _resolve_scope_chunk_size(config) == 128


def test_scope_chunk_size_can_disable_chunking(tmp_path):
    config = PipelineConfig(
        geojson_path=tmp_path / "field.geojson",
        start_date="2021-06-01",
        end_date="2021-06-02",
        crop_type="wheat",
        start_of_season=120,
        year=2021,
        scope_chunk_size=0,
    )

    assert _resolve_scope_chunk_size(config) is None


def test_pipeline_config_still_allows_fluorescence_scope_workflow(tmp_path):
    config = PipelineConfig(
        geojson_path=tmp_path / "field.geojson",
        start_date="2021-06-01",
        end_date="2021-06-02",
        crop_type="wheat",
        start_of_season=120,
        year=2021,
        scope_workflow="fluorescence",
    )

    assert config.resolved_scope_options["calc_fluor"] == 1
    assert config.resolved_scope_options["calc_planck"] == 0
