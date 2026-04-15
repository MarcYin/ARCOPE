"""Convert ARC retrieval outputs to SCOPE-compatible xarray DataArrays.

Two entry paths:

1. ``arc_arrays_to_scope_inputs`` — from live ``arc_field()`` return values
2. ``arc_npz_to_scope_inputs``   — from a saved NPZ file on disk

Both produce the ``(post_bio_da, post_bio_scale_da)`` tuple that
``scope.io.prepare.prepare_scope_input_dataset`` expects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from arc_scope.bridge.parameter_map import BIO_BANDS, BIO_SCALES, SCALE_BANDS


def arc_arrays_to_scope_inputs(
    post_bio_tensor: np.ndarray,
    scale_data: np.ndarray,
    mask: np.ndarray,
    doys: np.ndarray,
    geotransform: np.ndarray,
    crs: Any,
    year: int,
    *,
    post_bio_unc_tensor: np.ndarray | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Build SCOPE-ready xarray DataArrays from live ARC ``arc_field()`` outputs.

    Parameters
    ----------
    post_bio_tensor:
        Posterior biophysical parameters, shape ``(n_valid_pixels, 7, n_times)``.
        Integer-coded values (before applying BIO_SCALES).
    scale_data:
        Concatenated scale parameters, shape ``(n_valid_pixels, 15)``.
        Columns: 7 bio scale + 4 phenology + 4 soil.
    mask:
        Boolean mask, shape ``(ny, nx)``.  ``True`` means *masked out*.
    doys:
        1-D array of day-of-year values, length ``n_times``.
    geotransform:
        GDAL-style geotransform ``[x_origin, x_size, x_rot, y_origin, y_rot, y_size]``.
    crs:
        Coordinate reference system (string, EPSG code, or WKT).
    year:
        Calendar year to combine with *doys* for datetime coordinates.
    post_bio_unc_tensor:
        Optional uncertainty tensor (kept for provenance, not used by SCOPE).

    Returns
    -------
    post_bio_da:
        ``xr.DataArray`` with dims ``(y, x, band, time)`` and physical units.
    post_bio_scale_da:
        ``xr.DataArray`` with dims ``(y, x, band)`` containing per-pixel
        scale / soil / phenology parameters.
    """
    ny, nx = mask.shape[:2]
    n_valid = int((~mask).sum())
    nt = int(doys.size)

    post_bio_tensor = np.asarray(post_bio_tensor, dtype=np.float64)
    scale_data = np.asarray(scale_data, dtype=np.float64)

    # Reconstruct full spatial grids from valid-pixel-only arrays
    bio_full = np.full((ny, nx, len(BIO_BANDS), nt), np.nan, dtype=np.float64)
    bio_full[~mask] = post_bio_tensor.reshape(n_valid, len(BIO_BANDS), nt)

    scale_full = np.full((ny, nx, len(SCALE_BANDS)), np.nan, dtype=np.float64)
    scale_full[~mask] = scale_data.reshape(n_valid, len(SCALE_BANDS))

    # Build coordinate arrays
    geotransform = np.asarray(geotransform, dtype=np.float64)
    x_coords = geotransform[0] + np.arange(nx) * geotransform[1]
    y_coords = geotransform[3] + np.arange(ny) * geotransform[5]

    times = pd.to_datetime(
        [f"{year}{int(doy):03d}" for doy in doys], format="%Y%j"
    )
    _, unique_index = np.unique(times.values, return_index=True)
    unique_index = np.sort(unique_index)

    # Apply scale factors to convert integer-coded values to physical units
    scales = np.asarray(BIO_SCALES, dtype=np.float64).reshape(1, 1, len(BIO_BANDS), 1)

    post_bio_da = xr.DataArray(
        bio_full[:, :, :, unique_index] * scales,
        dims=("y", "x", "band", "time"),
        coords={
            "y": y_coords,
            "x": x_coords,
            "band": list(BIO_BANDS),
            "time": times[unique_index],
        },
        name="post_bio",
    )
    post_bio_scale_da = xr.DataArray(
        scale_full,
        dims=("y", "x", "band"),
        coords={
            "y": y_coords,
            "x": x_coords,
            "band": list(SCALE_BANDS),
        },
        name="post_bio_scale",
    )

    # Write spatial CRS if rioxarray is available
    post_bio_da = _write_crs(post_bio_da, crs)
    post_bio_scale_da = _write_crs(post_bio_scale_da, crs)

    return post_bio_da, post_bio_scale_da


def arc_npz_to_scope_inputs(
    npz_path: str | Path,
    year: int,
    *,
    reference_dataset: xr.Dataset | xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Load ARC outputs from a saved NPZ file and convert to SCOPE inputs.

    This delegates to SCOPE's ``read_s2_bio_inputs`` when available, falling
    back to a standalone implementation otherwise.

    Parameters
    ----------
    npz_path:
        Path to the ``.npz`` file saved by ``arc_field()``.
    year:
        Calendar year for datetime coordinate construction.
    reference_dataset:
        Optional reference for spatial CRS alignment.

    Returns
    -------
    post_bio_da, post_bio_scale_da:
        Same format as :func:`arc_arrays_to_scope_inputs`.
    """
    try:
        from scope.io.prepare import read_s2_bio_inputs

        return read_s2_bio_inputs(
            npz_path, year=year, reference_dataset=reference_dataset
        )
    except ImportError:
        pass

    # Standalone fallback: replicate the NPZ loading logic
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as payload:
        doys = np.asarray(payload["doys"])
        mask = np.asarray(payload["mask"])
        geotransform = np.asarray(payload["geotransform"], dtype=np.float64)
        post_bio_tensor = np.asarray(payload["post_bio_tensor"], dtype=np.float64)
        scale_data = np.asarray(payload["dat"], dtype=np.float64)
        crs_value = payload.get("crs")

    crs = _normalise_crs(crs_value)

    ny, nx = mask.shape[:2]
    nt = int(doys.size)

    # The NPZ may store data in flattened (n_valid, ...) or full (ny*nx, ...) form.
    expected_full = ny * nx * len(BIO_BANDS) * nt
    if post_bio_tensor.size == expected_full:
        # Full grid format: reshape directly
        bio_grid = post_bio_tensor.reshape(ny, nx, len(BIO_BANDS), nt)
        scale_grid = scale_data.reshape(ny, nx, len(SCALE_BANDS))
    else:
        # Flattened valid-pixel format: reconstruct grid
        n_valid = int((~mask).sum())
        bio_grid = np.full((ny, nx, len(BIO_BANDS), nt), np.nan, dtype=np.float64)
        bio_grid[~mask] = post_bio_tensor.reshape(n_valid, len(BIO_BANDS), nt)
        scale_grid = np.full((ny, nx, len(SCALE_BANDS)), np.nan, dtype=np.float64)
        scale_grid[~mask] = scale_data.reshape(n_valid, len(SCALE_BANDS))

    times = pd.to_datetime(
        [f"{year}{int(doy):03d}" for doy in doys], format="%Y%j"
    )
    _, unique_index = np.unique(times.values, return_index=True)
    unique_index = np.sort(unique_index)

    x_coords = geotransform[0] + np.arange(nx) * geotransform[1]
    y_coords = geotransform[3] + np.arange(ny) * geotransform[5]
    scales = np.asarray(BIO_SCALES, dtype=np.float64).reshape(1, 1, len(BIO_BANDS), 1)

    post_bio_da = xr.DataArray(
        bio_grid[:, :, :, unique_index] * scales,
        dims=("y", "x", "band", "time"),
        coords={
            "y": y_coords,
            "x": x_coords,
            "band": list(BIO_BANDS),
            "time": times[unique_index],
        },
        name="post_bio",
    )
    post_bio_scale_da = xr.DataArray(
        scale_grid,
        dims=("y", "x", "band"),
        coords={
            "y": y_coords,
            "x": x_coords,
            "band": list(SCALE_BANDS),
        },
        name="post_bio_scale",
    )

    post_bio_da = _write_crs(post_bio_da, crs)
    post_bio_scale_da = _write_crs(post_bio_scale_da, crs)

    return post_bio_da, post_bio_scale_da


def _write_crs(data: xr.DataArray, crs: Any) -> xr.DataArray:
    """Attach CRS metadata via rioxarray if available."""
    if crs is None:
        return data
    try:
        import rioxarray  # noqa: F401

        data = data.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
        return data.rio.write_crs(crs, inplace=False)
    except Exception:
        return data


def _normalise_crs(value: Any) -> Any:
    """Decode bytes CRS values from NPZ files."""
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 0:
        scalar = array.item()
        if isinstance(scalar, bytes):
            return scalar.decode("utf-8")
        return scalar
    return value
