# Changelog

All notable changes to arcope are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2026-04-28

### Fixed
- Corrected the scipy autograd release by adding a tensor-preserving SCOPE
  objective path for built-in `ArcScopePipeline` optimisation. Gradient
  evaluations now call SCOPE's raw tensor-returning workflow methods and compute
  loss before xarray/NetCDF dataset assembly can detach tensors.
- Built-in SCOPE pipeline optimisation now defaults `use_autograd_jac` to
  `"required"` so production runs fail loudly if the forward graph detaches.
  Custom proxy runners keep the compatibility default of `"auto"`.
- `ArcScopePipeline.run_optimization()` now uses the built-in optimisation
  runners directly instead of passing a custom lambda that disabled the
  tensor-preserving autograd branch.
- The standalone thermal preset now fits the prescribed thermal variables
  (`Tcu`, `Tch`, `Tsu`, `Tsh`); resistance terms (`rss`, `rbs`) remain part of
  the coupled energy-balance preset.

## [0.1.2] - 2026-04-28

### Added
- `ScipyOptimizer` now supplies an autograd-backed `jac` callable to scipy when
  the objective can provide PyTorch gradients, avoiding finite-difference
  parameter probing for differentiable SCOPE runs.

## [0.1.1] - 2026-04-27

### Added
- `ArcScopePipeline.run()` now honours `PipelineConfig.optimize` and
  `optim_config["enabled"]` / nested `optim.enabled` payloads by running a real
  `ScopeObjective` + optimiser loop before the final SCOPE simulation.
- Added `OptimizationResult` metadata to `PipelineResult` and
  `arc_scope_optimization_*` dataset attrs so downstream manifests can
  distinguish optimised runs from plain simulations.
- Added pipeline optimisation configuration docs covering observations,
  target variables, parameter specs, workflow defaults, and scipy optimiser
  settings.
- Added an optimisation guide with fluorescence, thermal, coupled
  energy-balance, and fixed-parameter fitting examples plus interpretation of
  optimisation outputs.
- Added reproducible optimisation example artifacts, including generated JSON,
  CSV, and SVG visualisations for SIF, thermal, and coupled energy-balance
  fits.

### Fixed
- Optimisation-enabled runner submissions now fail loudly when observed target
  data is missing instead of returning a successful unoptimised simulation.
- `ScopeObjective.evaluate()` now raises when requested target variables are
  absent or no finite prediction/observation pairs exist, preventing silent
  zero-loss optimisation results.

### Changed
- Moved pipeline optimisation wiring into `arc_scope.pipeline.optimization`
  while keeping `ArcScopePipeline.run_optimization()` as the runner-facing entry
  point.
- Added an explicit source distribution manifest so docs/examples/tests ship
  with release tarballs while heavyweight generated NetCDF/NPZ intermediates
  stay out of the package.

## [0.1.0] - 2026-04-18

### Added
- Initial public release.
- **Bridge layer** (`arc_scope.bridge`): ARC-to-SCOPE parameter mapping with
  live-array and NPZ entry points, 7-bio + 4-soil SCOPE variable mapping,
  and BSM soil validation.
- **Weather providers** (`arc_scope.weather`): pluggable `WeatherProvider` ABC,
  ERA5 via CDS API with disk caching and month-chunked downloads, local
  CSV/NetCDF provider, BRL diffuse-fraction radiation partitioning, and
  CF-compliant NetCDF3 serialisation for scipy engine compatibility.
- **Pipeline orchestration** (`arc_scope.pipeline`): `PipelineConfig`
  dataclass with four workflows (reflectance, fluorescence, thermal,
  energy-balance), `ArcScopePipeline` runner, and composable step
  functions.
- **Optimisation extension** (`arc_scope.optim`): `ParameterSet` with
  bounded transforms (log, logit), `ScopeObjective` wrapper, and
  pluggable optimiser protocol with scipy + PyTorch wrappers.
- **Utilities**: Spencer-based solar geometry, GeoJSON loaders, shared
  type aliases.
- **Experiments**:
  - `arc_scope.experiments.showcase`: core-dependency showcase with
    synthetic ARC retrieval, local weather, and proxy SIF calibration.
    Ships a single-scroll Plotly dashboard plus SVG charts.
  - `arc_scope.experiments.dual_workflow`: real end-to-end ARC -> SCOPE
    experiment running reflectance, fluorescence, and thermal workflows
    from one shared retrieval. Emits an interactive Plotly explorer
    with 5 tabs (overview / spatial / temporal / spectral / compare),
    animation, colorscale lock, click-to-jump, NDVI derived indices,
    CSV export, and URL state persistence.
- **Bundled data**: Belgium test field GeoJSON and showcase weather CSV
  covering May 15 – Oct 7.
- **Documentation**: MkDocs Material site deployed at
  [marcyin.github.io/ARCOPE](https://marcyin.github.io/ARCOPE/) with
  installation, quickstart, architecture, API reference, showcase, and
  full-run example pages.
- **Tests**: 103 tests covering bridge, weather, pipeline, optim, utils,
  experiments, and API.
- **CI**: GitHub Actions for tests (Python 3.9-3.14), docs build + deploy,
  and PyPI release via OIDC trusted publishing.

[Unreleased]: https://github.com/MarcYin/ARCOPE/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/MarcYin/ARCOPE/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/MarcYin/ARCOPE/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/MarcYin/ARCOPE/releases/tag/v0.1.0
