# Changelog

All notable changes to arcope are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.10] - 2026-05-11

### Fixed
- `ParameterSet.inject_into_dataset()` now adds missing optimisation parameters
  on the SCOPE `y/x/time` grid when available, matching the tensor-preserving
  optimisation injection paths used before the final forward run.
- Streaming PyTorch MSE gradients now backpropagate each yielded SCOPE chunk
  before advancing the chunk iterator, avoiding autograd version-counter
  failures when upstream chunk outputs share optics-path views.
- `prepare_scope_dataset()` now seeds energy-balance defaults for explicit
  `calc_ebal=1` configurations and retries SCOPE input preparation when
  upstream validation rejects variables that ARC-SCOPE derives locally.

### Changed
- Raised the optional SCOPE dependency floor to `scope-rtm>=0.4.5`, the release
  with the small-grid energy-balance fast path needed by optimisation runs.

## [0.1.9] - 2026-05-05

### Fixed
- `prepare_scope_dataset()` now broadcasts time-indexed observation geometry
  variables, such as `tts`, `tto`, and `psi`, onto the SCOPE `y/x/time` grid
  before calling `scope-rtm`.
- Tensor-preserving SCOPE runners now seed missing optimisation parameter
  variables with SCOPE-grid placeholders instead of scalar fields, allowing the
  tensor data module to broadcast the active parameter values during batches.

## [0.1.8] - 2026-05-04

### Fixed
- `prepare_scope_dataset()` now explicitly broadcasts time-indexed field-scale
  weather variables onto the SCOPE `y/x/time` grid before handing data to
  `scope-rtm`.
- `ScopeObjective._inject_params()` now adds missing optimisation parameter
  fields with the base dataset's `y/x/time` shape instead of scalar variables.

### Changed
- Raised the optional SCOPE dependency floor to `scope-rtm>=0.4.4`, which fixes
  upstream datetime/timedelta averaging in prepared SCOPE input datasets.

## [0.1.7] - 2026-05-04

### Fixed
- `build_observation_dataset()` now honours the documented Sentinel-2
  local-solar-time overpass contract before computing solar geometry, avoiding
  large UTC/LST shifts at western longitudes.
- Streaming PyTorch MSE gradients now avoid retaining every chunk graph during
  backward propagation, reducing autograd memory growth while preserving the
  parameter-transform chain rule.
- The SCOPE output roundtrip test now accepts both SciPy NetCDF3 fallback
  output and NetCDF4 output from SCOPE's exporter.

### Changed
- Raised the optional SCOPE dependency floor to `scope-rtm>=0.4.3` and updated
  the development lockfile/runtime target accordingly.

## [0.1.6] - 2026-05-03

### Fixed
- `ScopeObjective` now aligns predictions and observations by xarray
  coordinates instead of truncating flattened arrays, preventing gridded
  SCOPE output from being compared against the corner pixel for point
  observations.
- The numpy, PyTorch, streaming forward, and streaming backward objective
  paths now share the same coordinate-alignment contract and raise when
  aligned coordinates have no overlap.

### Added
- Added `pixel_selector` support for point-observation optimisation against
  gridded predictions, including coordinate-label and positional-index forms.

## [0.1.5] - 2026-05-02

### Fixed
- Updated differentiable SCOPE optimisation to use upstream `scope-rtm`
  streaming tensor chunks and backpropagate each chunk loss immediately, so
  `scope_chunk_size` can reduce peak autograd memory instead of retaining all
  chunk graphs until one full-batch backward pass.

### Changed
- Raised the optional SCOPE dependency to `scope-rtm>=0.4.2`, the release that
  exposes the streaming tensor iterators used by ARC-SCOPE.

## [0.1.4] - 2026-04-29

### Added
- Added `PipelineConfig.scope_chunk_size` and CLI `--scope-chunk-size` so
  ARC-SCOPE passes SCOPE's native `SimulationConfig.chunk_size` through normal
  forward runs and tensor-preserving optimisation evaluations.
- Optimisation configs can override the pipeline chunk size with
  `scope_chunk_size`, `chunk_size`, or `batch_size`, including nested
  `optim` runner payloads.

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

[Unreleased]: https://github.com/MarcYin/ARCOPE/compare/v0.1.8...HEAD
[0.1.8]: https://github.com/MarcYin/ARCOPE/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/MarcYin/ARCOPE/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/MarcYin/ARCOPE/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/MarcYin/ARCOPE/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/MarcYin/ARCOPE/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/MarcYin/ARCOPE/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/MarcYin/ARCOPE/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/MarcYin/ARCOPE/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/MarcYin/ARCOPE/releases/tag/v0.1.0
