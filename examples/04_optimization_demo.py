"""Example 04: Parameter optimisation workflow.

This script demonstrates the optimisation module: creating parameter sets,
working with reparameterisation transforms, injecting parameters into
datasets, and the conceptual optimisation workflow.

Requirements:
    pip install arc-scope                # core (for ParameterSet, transforms)
    pip install "arc-scope[optim]"       # for TorchOptimizer
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from arc_scope.optim.parameters import (
    ENERGY_BALANCE_OPTIMIZATION_PARAMS,
    SIF_OPTIMIZATION_PARAMS,
    THERMAL_OPTIMIZATION_PARAMS,
    ParameterSet,
    ParameterSpec,
)


def main() -> None:
    print("=" * 60)
    print("ARC-SCOPE Optimisation Demo")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Create a ParameterSet
    # ------------------------------------------------------------------
    print("\n--- Step 1: Create a ParameterSet ---")

    params = ParameterSet([
        ParameterSpec(
            name="fqe",
            initial=0.01,
            lower=0.001,
            upper=0.1,
            optimize=True,
            transform="log",      # log-space for strictly positive params
        ),
        ParameterSpec(
            name="rss",
            initial=500.0,
            lower=10.0,
            upper=5000.0,
            optimize=True,
            transform="log",
        ),
        ParameterSpec(
            name="rwc",
            initial=0.5,
            lower=0.1,
            upper=1.0,
            optimize=True,
            transform="logit",    # logit for [0,1]-bounded params
        ),
        ParameterSpec(
            name="Cd",
            initial=0.2,
            lower=0.01,
            upper=1.0,
            optimize=False,       # held fixed during optimisation
            transform="identity",
        ),
    ])

    print(f"Total parameters:      {len(params.specs)}")
    print(f"Optimisable:           {len(params.optimizable)}")
    print(f"Fixed:                 {len(params.fixed)}")

    for spec in params.specs:
        status = "optimise" if spec.optimize else "fixed"
        print(f"  {spec.name:>5s}: initial={spec.initial:<8g}  "
              f"bounds=[{spec.lower}, {spec.upper}]  "
              f"transform={spec.transform:<8s}  ({status})")

    # ------------------------------------------------------------------
    # Step 2: Reparameterisation transforms
    # ------------------------------------------------------------------
    print("\n--- Step 2: Transform round-trips ---")
    print("Transforms map physical values to unconstrained optimisation space:")

    for spec in params.optimizable:
        physical = spec.initial
        unconstrained = spec.to_unconstrained(physical)
        recovered = spec.to_physical(unconstrained)
        print(f"  {spec.name:>5s}: physical={physical:<10g} -> "
              f"unconstrained={unconstrained:<12.6f} -> "
              f"recovered={recovered:<10g}  "
              f"(match: {np.isclose(physical, recovered)})")

    # ------------------------------------------------------------------
    # Step 3: Array conversion for optimisers
    # ------------------------------------------------------------------
    print("\n--- Step 3: Array conversion ---")

    # to_array() returns unconstrained values for optimisable params only
    x = params.to_array()
    print(f"Unconstrained vector: {x}")
    print(f"Shape: {x.shape} (one per optimisable param)")

    # from_array() maps back to named physical values (all params)
    named = params.from_array(x)
    print(f"Named physical values: {named}")

    # Perturb and recover
    x_perturbed = x + 0.5
    named_perturbed = params.from_array(x_perturbed)
    print(f"\nPerturbed (+0.5 in unconstrained space):")
    for name, val in named_perturbed.items():
        original = named[name]
        print(f"  {name:>5s}: {original:.6f} -> {val:.6f}")

    # ------------------------------------------------------------------
    # Step 4: Inject parameters into an xr.Dataset
    # ------------------------------------------------------------------
    print("\n--- Step 4: Dataset injection ---")

    # Create a minimal synthetic SCOPE-like dataset
    times = np.array(["2021-07-01", "2021-07-15", "2021-08-01"], dtype="datetime64")
    synthetic_ds = xr.Dataset(
        {
            "LAI": (("y", "x", "time"), np.full((4, 4, 3), 3.0)),
            "Cab": (("y", "x", "time"), np.full((4, 4, 3), 40.0)),
        },
        coords={
            "y": np.arange(4),
            "x": np.arange(4),
            "time": times,
        },
    )
    print(f"Original dataset variables: {list(synthetic_ds.data_vars)}")

    # Inject parameter values
    injected = params.inject_into_dataset(synthetic_ds)
    print(f"After injection:           {list(injected.data_vars)}")
    for spec in params.specs:
        if spec.name in injected:
            val = float(injected[spec.name].values) if injected[spec.name].ndim == 0 else "broadcast"
            print(f"  {spec.name:>5s} = {val}")

    # Inject with custom values
    custom_values = {"fqe": 0.05, "rss": 200.0, "rwc": 0.8, "Cd": 0.2}
    injected2 = params.inject_into_dataset(synthetic_ds, values=custom_values)
    print(f"\nWith custom values:")
    for name, val in custom_values.items():
        print(f"  {name:>5s} = {val}")

    # ------------------------------------------------------------------
    # Step 5: Pre-configured parameter sets
    # ------------------------------------------------------------------
    print("\n--- Step 5: Pre-configured parameter sets ---")

    for label, pset in [
        ("SIF_OPTIMIZATION_PARAMS", SIF_OPTIMIZATION_PARAMS),
        ("THERMAL_OPTIMIZATION_PARAMS", THERMAL_OPTIMIZATION_PARAMS),
        ("ENERGY_BALANCE_OPTIMIZATION_PARAMS", ENERGY_BALANCE_OPTIMIZATION_PARAMS),
    ]:
        names = [s.name for s in pset.specs]
        print(f"  {label}:")
        print(f"    parameters: {names}")

    # ------------------------------------------------------------------
    # Step 6: PyTorch integration
    # ------------------------------------------------------------------
    print("\n--- Step 6: PyTorch integration ---")
    try:
        tensor = params.to_torch(device="cpu", dtype="float64")
        print(f"PyTorch tensor: {tensor}")
        print(f"  shape:         {tensor.shape}")
        print(f"  requires_grad: {tensor.requires_grad}")
        print(f"  dtype:         {tensor.dtype}")
    except ImportError:
        print("PyTorch is not installed.")
        print("Install with: pip install \"arc-scope[optim]\"")

    # ------------------------------------------------------------------
    # Step 7: Conceptual optimisation workflow
    # ------------------------------------------------------------------
    print("\n--- Step 7: Optimisation workflow (conceptual) ---")
    print("""
The full optimisation workflow:

  1. Run ARC retrieval to get biophysical parameters (held fixed).
  2. Prepare a SCOPE input dataset via the bridge + weather modules.
  3. Define a ParameterSet with parameters to optimise.
  4. Create observed data (satellite SIF, LST, or flux tower).
  5. Build a ScopeObjective:

     from arc_scope.optim.objective import ScopeObjective

     objective = ScopeObjective(
         base_dataset=scope_ds,
         observations=observed_sif,
         target_variables=["F685", "F740"],
         config=config,
     )

  6. Choose an optimiser:

     from arc_scope.optim.protocols import ScipyOptimizer
     optimizer = ScipyOptimizer(method="L-BFGS-B", max_iter=50)

  7. Run optimisation:

     optimised_params = optimizer.step(objective, params)

  8. The optimised ParameterSet has updated initial values.
""")

    print("Done.")


if __name__ == "__main__":
    main()
