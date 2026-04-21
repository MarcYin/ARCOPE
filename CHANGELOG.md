# Changelog

All notable changes to arcope are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/MarcYin/ARCOPE/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/MarcYin/ARCOPE/releases/tag/v0.1.0
