# Limits and Roadmap

## Verified Today

The repo currently has the strongest in-repo evidence for:

- bridge conversion from ARC-like arrays to SCOPE-style xarray inputs
- field bounds and centroid extraction
- observation geometry generation
- local weather ingestion
- shortwave radiation partitioning
- parameter container and scipy-based optimisation mechanics

## Optional, Not Centered in the Showcase

- ARC retrieval from Sentinel-2 data
- SCOPE dataset preparation and execution
- ERA5 downloads from CDS
- torch-based surrogate optimisation

These paths exist in the codebase, but the docs site does not use the showcase page to claim they are fully validated end to end inside this repository.

## Current Limits

- Full `scope-rtm` execution is an integration path, not the default verified onboarding story.
- The proxy calibration example demonstrates optimisation mechanics, not validated SCOPE inversion.
- ERA5 workflows require external credentials and network access.

## Near-Term Roadmap

- Add direct tests around `prepare_scope_dataset()` and `run_scope_simulation()`.
- Expand the showcase into an optional verified `scope-rtm` integration example once the dependency boundary is exercised in CI.
- Add more packaged example inputs and benchmark outputs for regression checks.
