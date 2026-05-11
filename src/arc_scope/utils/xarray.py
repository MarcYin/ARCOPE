"""Shared xarray helpers for SCOPE-shaped datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr


def dataset_grid_template(dataset: xr.Dataset) -> xr.DataArray | None:
    grid_dims = tuple(dim for dim in ("y", "x", "time") if dim in dataset.sizes)
    if grid_dims != ("y", "x", "time"):
        return None

    coords = {
        dim: dataset.coords[dim]
        for dim in grid_dims
        if dim in dataset.coords
    }
    shape = tuple(int(dataset.sizes[dim]) for dim in grid_dims)
    return xr.DataArray(np.empty(shape, dtype=np.float64), dims=grid_dims, coords=coords)


def dataarray_like(template: xr.DataArray, value: Any) -> xr.DataArray:
    shape = tuple(template.shape)
    raw = np.asarray(value)
    if shape:
        data = np.broadcast_to(raw, shape).copy()
    else:
        data = raw.reshape(())
    return xr.DataArray(
        data,
        dims=template.dims,
        coords=template.coords,
        attrs=template.attrs,
        name=template.name,
    )
