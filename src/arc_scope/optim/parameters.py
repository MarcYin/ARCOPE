"""Typed, bounds-aware parameter containers for SCOPE optimisation.

Design goals:
- Clean interface for gradient-based and gradient-free optimisers
- Reparameterisation transforms for bounded parameters
- Easy injection of current values into SCOPE's xarray input dataset
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import xarray as xr


@dataclass
class ParameterSpec:
    """Specification for a single optimisable parameter.

    Parameters
    ----------
    name:
        SCOPE variable name (e.g., ``"fqe"``, ``"rss"``).
    initial:
        Starting value in physical units.
    lower:
        Lower bound in physical units.
    upper:
        Upper bound in physical units.
    optimize:
        Whether this parameter is optimised (``True``) or held fixed.
    transform:
        Reparameterisation for unconstrained optimisation:
        - ``"identity"``: no transform
        - ``"log"``: log-space (for strictly positive parameters)
        - ``"logit"``: logit-space (for [0,1]-bounded parameters)
    """

    name: str
    initial: float
    lower: float
    upper: float
    optimize: bool = True
    transform: str = "identity"

    def to_unconstrained(self, value: float) -> float:
        """Map a physical value to unconstrained space."""
        if self.transform == "identity":
            return value
        elif self.transform == "log":
            return np.log(max(value, 1e-30))
        elif self.transform == "logit":
            # Normalise to [0, 1] then logit
            p = (value - self.lower) / (self.upper - self.lower)
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return np.log(p / (1 - p))
        raise ValueError(f"Unknown transform: {self.transform}")

    def to_physical(self, unconstrained: float) -> float:
        """Map an unconstrained value back to physical units."""
        if self.transform == "identity":
            return np.clip(unconstrained, self.lower, self.upper)
        elif self.transform == "log":
            return np.clip(np.exp(unconstrained), self.lower, self.upper)
        elif self.transform == "logit":
            p = 1.0 / (1.0 + np.exp(-unconstrained))
            return self.lower + p * (self.upper - self.lower)
        raise ValueError(f"Unknown transform: {self.transform}")


@dataclass
class ParameterSet:
    """Collection of parameters for SCOPE optimisation.

    Manages the mapping between a flat optimisation vector (in
    unconstrained space) and named SCOPE parameters (in physical units).
    """

    specs: list[ParameterSpec] = field(default_factory=list)

    @property
    def optimizable(self) -> list[ParameterSpec]:
        """Specs with ``optimize=True``."""
        return [s for s in self.specs if s.optimize]

    @property
    def fixed(self) -> list[ParameterSpec]:
        """Specs with ``optimize=False``."""
        return [s for s in self.specs if not s.optimize]

    def to_array(self) -> np.ndarray:
        """Current values as an unconstrained 1-D array (optimisable only)."""
        return np.array([s.to_unconstrained(s.initial) for s in self.optimizable])

    def from_array(self, values: np.ndarray) -> dict[str, float]:
        """Convert unconstrained array back to named physical values.

        Returns all parameters (optimised + fixed).
        """
        opt_specs = self.optimizable
        if len(values) != len(opt_specs):
            raise ValueError(
                f"Expected {len(opt_specs)} values, got {len(values)}"
            )

        result = {}
        for spec, val in zip(opt_specs, values):
            result[spec.name] = spec.to_physical(float(val))
        for spec in self.fixed:
            result[spec.name] = spec.initial
        return result

    def to_torch(self, device: str = "cpu", dtype: str = "float64"):
        """Create a PyTorch tensor with ``requires_grad`` for optimisable params.

        Returns
        -------
        torch.Tensor of shape ``(n_optimisable,)`` in unconstrained space.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for gradient-based optimisation")

        dtype_map = {"float32": torch.float32, "float64": torch.float64}
        values = self.to_array()
        return torch.tensor(
            values, dtype=dtype_map.get(dtype, torch.float64),
            device=device, requires_grad=True,
        )

    def inject_into_dataset(
        self,
        dataset: xr.Dataset,
        values: dict[str, float] | None = None,
    ) -> xr.Dataset:
        """Write current parameter values into an xr.Dataset.

        Parameters
        ----------
        dataset:
            The SCOPE input dataset to modify.
        values:
            Named parameter values in physical units. If ``None``,
            uses the initial values.

        Returns
        -------
        Modified dataset with parameter values set.
        """
        if values is None:
            values = {s.name: s.initial for s in self.specs}

        ds = dataset.copy(deep=True)
        for name, val in values.items():
            if name in ds:
                ds[name] = ds[name] * 0 + val  # Broadcast scalar to existing shape
            else:
                # Add as a new scalar variable
                ds[name] = val
        return ds


# Pre-configured parameter sets for common optimisation scenarios

SIF_OPTIMIZATION_PARAMS = ParameterSet([
    ParameterSpec("fqe", initial=0.01, lower=0.001, upper=0.1, transform="log"),
])

THERMAL_OPTIMIZATION_PARAMS = ParameterSet([
    ParameterSpec("rss", initial=500.0, lower=10.0, upper=5000.0, transform="log"),
    ParameterSpec("rbs", initial=10.0, lower=1.0, upper=100.0, transform="log"),
])

ENERGY_BALANCE_OPTIMIZATION_PARAMS = ParameterSet([
    ParameterSpec("fqe", initial=0.01, lower=0.001, upper=0.1, transform="log"),
    ParameterSpec("rss", initial=500.0, lower=10.0, upper=5000.0, transform="log"),
    ParameterSpec("rbs", initial=10.0, lower=1.0, upper=100.0, transform="log"),
    ParameterSpec("Cd", initial=0.2, lower=0.01, upper=1.0, transform="log"),
    ParameterSpec("rwc", initial=0.5, lower=0.1, upper=1.0, transform="logit"),
])
