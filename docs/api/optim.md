# API Reference: `arc_scope.optim`

The optimisation module provides parameter tuning for SCOPE simulations. It adjusts parameters that cannot be retrieved from reflectance (e.g., fluorescence quantum efficiency, soil resistance) while holding ARC-retrieved biophysical parameters fixed.

## `ParameterSpec`

```python
@dataclass
class ParameterSpec:
    name: str
    initial: float
    lower: float
    upper: float
    optimize: bool = True
    transform: str = "identity"
```

Specification for a single optimisable parameter.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *required* | SCOPE variable name (e.g., `"fqe"`, `"rss"`). |
| `initial` | `float` | *required* | Starting value in physical units. |
| `lower` | `float` | *required* | Lower bound in physical units. |
| `upper` | `float` | *required* | Upper bound in physical units. |
| `optimize` | `bool` | `True` | Whether this parameter is optimised or held fixed. |
| `transform` | `str` | `"identity"` | Reparameterisation: `"identity"`, `"log"`, or `"logit"`. |

### Methods

**`to_unconstrained(value)`** -- Map a physical value to unconstrained optimisation space:

- `"identity"`: returns the value unchanged
- `"log"`: returns `log(value)` (for strictly positive parameters)
- `"logit"`: normalises to [0, 1] then applies logit (for bounded parameters)

**`to_physical(unconstrained)`** -- Inverse of `to_unconstrained()`. Maps back to physical units with clipping to bounds.

### Usage

```python
from arc_scope.optim.parameters import ParameterSpec

spec = ParameterSpec("fqe", initial=0.01, lower=0.001, upper=0.1, transform="log")

# Round-trip through unconstrained space
u = spec.to_unconstrained(0.01)   # -4.605...
p = spec.to_physical(u)           # 0.01
```

## `ParameterSet`

```python
@dataclass
class ParameterSet:
    specs: list[ParameterSpec] = field(default_factory=list)
```

Collection of parameters for SCOPE optimisation. Manages the mapping between a flat optimisation vector (unconstrained space) and named SCOPE parameters (physical units).

### Properties

**`optimizable`** -- List of `ParameterSpec` where `optimize=True`.

**`fixed`** -- List of `ParameterSpec` where `optimize=False`.

### Methods

**`to_array()`** -- Current values as an unconstrained 1-D numpy array (optimisable parameters only).

```python
params = ParameterSet([
    ParameterSpec("fqe", initial=0.01, lower=0.001, upper=0.1, transform="log"),
    ParameterSpec("rss", initial=500, lower=10, upper=5000, transform="log"),
])
x = params.to_array()  # shape (2,) in unconstrained space
```

**`from_array(values)`** -- Convert an unconstrained array back to a dict of named physical values. Returns all parameters (optimised + fixed).

```python
named = params.from_array(x)  # {"fqe": 0.01, "rss": 500.0}
```

**`to_torch(device="cpu", dtype="float64")`** -- Create a PyTorch tensor with `requires_grad=True` for gradient-based optimisation. Returns a tensor of shape `(n_optimisable,)` in unconstrained space. Requires `torch`.

**`inject_into_dataset(dataset, values=None)`** -- Write parameter values into an `xr.Dataset`. If `values` is `None`, uses the initial values from each spec. Existing variables are broadcast-updated; new variables are added as scalars.

```python
ds = params.inject_into_dataset(scope_dataset, {"fqe": 0.02, "rss": 300.0})
```

## Pre-configured Parameter Sets

The module provides ready-to-use `ParameterSet` instances:

### `SIF_OPTIMIZATION_PARAMS`

For SIF-focused optimisation:

```python
SIF_OPTIMIZATION_PARAMS = ParameterSet([
    ParameterSpec("fqe", initial=0.01, lower=0.001, upper=0.1, transform="log"),
])
```

### `THERMAL_OPTIMIZATION_PARAMS`

For thermal/LST optimisation:

```python
THERMAL_OPTIMIZATION_PARAMS = ParameterSet([
    ParameterSpec("rss", initial=500.0, lower=10.0, upper=5000.0, transform="log"),
    ParameterSpec("rbs", initial=10.0, lower=1.0, upper=100.0, transform="log"),
])
```

### `ENERGY_BALANCE_OPTIMIZATION_PARAMS`

For full energy-balance optimisation:

```python
ENERGY_BALANCE_OPTIMIZATION_PARAMS = ParameterSet([
    ParameterSpec("fqe", initial=0.01, lower=0.001, upper=0.1, transform="log"),
    ParameterSpec("rss", initial=500.0, lower=10.0, upper=5000.0, transform="log"),
    ParameterSpec("rbs", initial=10.0, lower=1.0, upper=100.0, transform="log"),
    ParameterSpec("Cd", initial=0.2, lower=0.01, upper=1.0, transform="log"),
    ParameterSpec("rwc", initial=0.5, lower=0.1, upper=1.0, transform="logit"),
])
```

## `ScopeObjective`

```python
class ScopeObjective:
    def __init__(
        self,
        base_dataset: xr.Dataset,
        observations: xr.Dataset,
        target_variables: Sequence[str],
        loss_fn: Callable | None = None,
        scope_runner: Callable | None = None,
        config: Any = None,
    ): ...
```

Objective function wrapping a SCOPE forward pass. Injects parameters into the prepared dataset, runs SCOPE, and computes a scalar loss against observations.

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `base_dataset` | `xr.Dataset` | The prepared SCOPE input dataset. |
| `observations` | `xr.Dataset` | Observed data to compare against (e.g., satellite SIF or LST). |
| `target_variables` | `Sequence[str]` | SCOPE output variable names to extract and compare. |
| `loss_fn` | `Callable` or `None` | Custom loss function `(predicted, observed) -> scalar`. Defaults to MSE. |
| `scope_runner` | `Callable` or `None` | Custom SCOPE runner. Defaults to `run_scope_simulation`. |
| `config` | `Any` | Pipeline configuration for SCOPE execution. |

### Methods

**`evaluate(params)`** -- Evaluate the objective (numpy/scipy-compatible). Takes a dict of named parameter values in physical units. Returns a scalar loss value.

**`evaluate_torch(params, param_tensor, param_set)`** -- Evaluate with PyTorch autograd support. Currently wraps `evaluate()` with a surrogate gradient.

## `Optimizer` Protocol

```python
@runtime_checkable
class Optimizer(Protocol):
    def step(self, objective: ScopeObjective, params: ParameterSet) -> ParameterSet: ...
    def converged(self) -> bool: ...
```

Protocol that any optimiser must satisfy for interchangeable use.

## `ScipyOptimizer`

```python
class ScipyOptimizer:
    def __init__(
        self,
        method: str = "L-BFGS-B",
        max_iter: int = 100,
        tol: float = 1e-6,
    ): ...
```

Wrapper around `scipy.optimize.minimize` for gradient-free optimisation.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `"L-BFGS-B"` | Scipy minimisation method. |
| `max_iter` | `100` | Maximum iterations. |
| `tol` | `1e-6` | Convergence tolerance. |

**`step(objective, params)`** runs a full scipy minimisation from the current parameter values and updates the specs with optimised values.

**`converged()`** returns `True` if the last `step()` call reported convergence.

## `TorchOptimizer`

```python
class TorchOptimizer:
    def __init__(
        self,
        optimizer_cls: type | None = None,
        lr: float = 0.01,
        max_steps: int = 100,
        tol: float = 1e-6,
        **optimizer_kwargs,
    ): ...
```

Wrapper around `torch.optim` optimisers for gradient-based optimisation. Requires `torch`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimizer_cls` | `None` (defaults to `Adam`) | A `torch.optim.Optimizer` class. |
| `lr` | `0.01` | Learning rate. |
| `max_steps` | `100` | Maximum gradient steps per call. |
| `tol` | `1e-6` | Convergence tolerance on loss change. |

**`step(objective, params)`** runs gradient descent from the current parameter values.

**`converged()`** returns `True` if the loss stopped improving within tolerance.
