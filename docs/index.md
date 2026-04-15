# ARC-SCOPE

ARC-SCOPE connects ARC-style crop parameter retrieval outputs with SCOPE-shaped experiment inputs.

The strongest verified path in this repository today is the **core-dependency showcase experiment**:

- synthetic ARC-like biophysical and soil arrays
- bundled field geometry
- local weather ingestion
- observation geometry
- direct/diffuse shortwave partitioning
- an optional proxy calibration step that demonstrates optimisation mechanics

This docs site does **not** use that showcase to claim a validated full `scope-rtm` run. Full ARC retrieval, SCOPE execution, and ERA5 downloads remain optional downstream integrations.

## Start Here

1. Read the [Showcase Experiment](showcase-experiment.md) page.
2. Follow the [Quick Start](quickstart.md) if you want the lower-level bridge entry point.
3. Use the [Installation Guide](installation.md) to add optional extras for ARC, SCOPE, or ERA5.

## Verified In-Repo

- `arc_scope.bridge.arc_arrays_to_scope_inputs`
- `arc_scope.pipeline.steps.build_observation_dataset`
- `arc_scope.weather.LocalProvider`
- `arc_scope.weather.radiation.partition_shortwave`
- `arc_scope.optim` parameter containers and scipy-based proxy fitting mechanics

## Optional Integrations

- ARC retrieval via `arc-scope[arc]`
- SCOPE preparation and execution via `arc-scope[scope]`
- ERA5 download workflows via `arc-scope[weather]`

## Primary Example

Run the showcase experiment after installing the core package:

```bash
pip install arc-scope
python3 -m arc_scope.experiments.showcase --output-dir ./showcase-output
```

The packaged module entry point writes a summary JSON file, a flat CSV file of per-date diagnostics, and two SVG charts you can inspect directly.
