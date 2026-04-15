# Architecture

This document describes the internal architecture of ARC-SCOPE, the data flow between modules, and the design decisions that underpin the system.

## Data Flow

```
  Input                       Module              Output
  -----                       ------              ------

  GeoJSON boundary    ───>  arc.arc_field()  ───>  post_bio_tensor  (n_valid, 7, n_times)
  Start/end dates                                  scale_data       (n_valid, 15)
  Crop type                                        mask             (ny, nx)
  Season parameters                                doys             (n_times,)
        |                                          geotransform     (6,)
        |                                          crs
        |
        v
  post_bio_tensor     ───>  bridge.arc_arrays_to_scope_inputs()
  scale_data                        |
  mask, doys, crs                   v
                            post_bio_da        xr.DataArray (y, x, band, time)
                            post_bio_scale_da  xr.DataArray (y, x, band)
        |
        v
  BBox + time range   ───>  weather.ERA5Provider.fetch()
  (or local file)           weather.LocalProvider.fetch()
                                    |
                                    v
                            weather_ds   xr.Dataset (time,) with Rin, Rli, Ta, ea, p, u
        |
        v
  doys + location     ───>  pipeline.steps.build_observation_dataset()
                                    |
                                    v
                            obs_ds   xr.Dataset (time,) with solar/viewing angles
        |
        v
  All datasets        ───>  scope.io.prepare.prepare_scope_input_dataset()
                                    |
                                    v
                            scope_input_ds   xr.Dataset ready for SCOPE runner
        |
        v
  scope_input_ds      ───>  scope.ScopeGridRunner.run_scope_dataset()
                                    |
                                    v
                            scope_output_ds  xr.Dataset with reflectance, SIF, fluxes, ...
        |
        v (optional)
  scope_output_ds     ───>  optim.ScopeObjective.evaluate()
  observations                      |
  ParameterSet                      v
                            Scalar loss  ───>  Optimizer.step()  ───>  Updated ParameterSet
```

## Module Responsibilities

### `arc_scope.bridge`

The bridge module handles the format conversion between ARC's raw output arrays and the named, scaled xarray DataArrays that SCOPE expects.

**Key responsibilities:**

- Apply `BIO_SCALES` to convert integer-coded ARC values to physical units
- Reconstruct full spatial grids from ARC's valid-pixel-only arrays using the boolean mask
- Build coordinate arrays from the GDAL geotransform
- Convert day-of-year arrays to datetime coordinates
- Attach CRS metadata via rioxarray (when available)
- Validate soil parameters against SCOPE's expected ranges

The bridge works without ARC or SCOPE installed -- it only depends on numpy, xarray, and pandas.

### `arc_scope.weather`

Provides meteorological forcing data through a pluggable provider interface.

**`WeatherProvider`** is the abstract base class. Concrete implementations:

- **`ERA5Provider`**: downloads ERA5 hourly reanalysis from the Copernicus Climate Data Store, converts units, and caches results on disk
- **`LocalProvider`**: loads a user-supplied CSV or NetCDF file with a column-name mapping to SCOPE variables

The `radiation` submodule partitions total incoming shortwave (`Rin`) into direct and diffuse components using the BRL diffuse-fraction model.

### `arc_scope.pipeline`

Orchestrates the end-to-end workflow with `ArcScopePipeline`. Each stage is exposed as a standalone step function for partial-pipeline use:

- `retrieve_arc()` -- run ARC retrieval
- `bridge_arc_to_scope()` -- convert ARC outputs
- `fetch_weather()` -- get meteorological data
- `build_observation_dataset()` -- compute solar/viewing geometry
- `prepare_scope_dataset()` -- merge all inputs for SCOPE
- `run_scope_simulation()` -- execute the SCOPE model

`PipelineConfig` centralises all settings and maps workflow names to SCOPE option flags.

### `arc_scope.optim`

Provides parameter optimisation for SCOPE parameters that cannot be retrieved from reflectance alone (e.g., fluorescence quantum efficiency, soil resistance).

**Design:**

- `ParameterSpec` defines a single parameter with bounds, initial value, and reparameterisation transform
- `ParameterSet` manages collections of parameters and the mapping between flat optimisation vectors and named SCOPE variables
- `ScopeObjective` wraps the SCOPE forward pass as a loss function
- `Optimizer` protocol allows both scipy and PyTorch optimisers

### `arc_scope.utils`

Shared helpers for geometry computations (`solar_position`, `relative_azimuth`), GeoJSON I/O (`load_geojson_bounds`, `load_geojson_centroid`), and type aliases.

## ARC to SCOPE Parameter Mapping

ARC's `post_bio_tensor` contains 7 biophysical parameters per pixel per timestep. The bridge maps these to SCOPE variable names with scale factors:

| ARC index | ARC name | SCOPE name | Scale factor | Physical units |
|:---------:|----------|------------|:------------:|----------------|
| 0 | N | N | 1/100 | dimensionless (leaf structure) |
| 1 | Cab | Cab | 1/100 | ug cm-2 (chlorophyll a+b) |
| 2 | Cm | Cdm | 1/10000 | g cm-2 (dry matter) |
| 3 | Cw | Cw | 1/10000 | g cm-2 (water content) |
| 4 | LAI | LAI | 1/100 | m2 m-2 (leaf area index) |
| 5 | ALA | ala | 1/100 | degrees (average leaf angle) |
| 6 | Cbrown | Cs | 1/1000 | dimensionless (senescent material) |

ARC's `scale_data` columns 11-14 provide soil BSM parameters:

| ARC index | SCOPE name | Physical range | Units |
|:---------:|------------|:--------------:|-------|
| 11 | BSMBrightness | 0.1 -- 0.7 | dimensionless |
| 12 | BSMlat | 10 -- 30 | degrees |
| 13 | BSMlon | 10 -- 70 | degrees |
| 14 | SMC | 2 -- 100 | % volumetric |

## Weather Variable Mapping

ERA5 variables are converted to SCOPE conventions:

| ERA5 variable | SCOPE variable | Unit conversion |
|---------------|:--------------:|-----------------|
| `2m_temperature` | `Ta` | K to degC |
| `2m_dewpoint_temperature` | `ea` | K to hPa (Magnus formula) |
| `surface_solar_radiation_downwards` | `Rin` | J m-2 to W m-2 |
| `surface_thermal_radiation_downwards` | `Rli` | J m-2 to W m-2 |
| `surface_pressure` | `p` | Pa to hPa |
| `10m_u/v_component_of_wind` | `u` | Vector magnitude (m s-1) |

## How the Optimisation Loop Works

1. ARC-retrieved biophysical parameters are held fixed throughout optimisation.
2. A `ParameterSet` defines which SCOPE parameters to optimise (e.g., `fqe`, `rss`), with bounds and reparameterisation transforms.
3. The `ScopeObjective` injects the current parameter values into the prepared SCOPE dataset, runs a forward simulation, and computes a scalar loss against observations.
4. An `Optimizer` (scipy or PyTorch) iterates:
   - Convert unconstrained vector to physical values via `ParameterSet.from_array()`
   - Inject into SCOPE dataset via `ParameterSet.inject_into_dataset()`
   - Run SCOPE forward pass
   - Compute loss (default: MSE between predicted and observed SIF/LST)
   - Update parameters
5. The optimised `ParameterSet` is returned with updated initial values.

Reparameterisation transforms enable unconstrained optimisation:

- **`identity`**: no transform, values clipped to bounds
- **`log`**: `x_unconstrained = log(x_physical)` for strictly positive parameters
- **`logit`**: normalise to [0,1] then logit, for bounded parameters

## JAX/PyTorch Boundary

SCOPE's simulation core runs on PyTorch (`scope-rtm` package). The bridge and weather modules are pure numpy/xarray and do not depend on PyTorch. The boundary is at `prepare_scope_dataset()` and `run_scope_simulation()`, which import from `scope` and `torch`.

This separation means:

- Bridge conversion, weather fetching, and data preparation work without PyTorch
- SCOPE simulation requires `torch >= 2.2` and `scope-rtm >= 0.2`
- The optimisation module bridges both worlds: `ScopeObjective.evaluate()` is numpy-compatible, while `evaluate_torch()` provides a PyTorch-autograd-compatible interface

The design allows lightweight use cases (bridge-only, data inspection) without heavy GPU dependencies, while supporting full differentiable simulation when all components are installed.
