"""Shared test fixtures for arc_scope tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def sample_mask():
    """A 10x10 boolean mask with some pixels masked out."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, :] = True  # First row masked
    mask[-1, :] = True  # Last row masked
    return mask


@pytest.fixture
def sample_doys():
    """Sample day-of-year array."""
    return np.array([150, 160, 170, 180, 190, 200])


@pytest.fixture
def sample_geotransform():
    """GDAL-style geotransform for a 10x10 grid."""
    return np.array([10.0, 0.001, 0.0, 50.0, 0.0, -0.001])


@pytest.fixture
def sample_arc_outputs(sample_mask, sample_doys, sample_geotransform):
    """Synthetic ARC field_processor outputs."""
    n_valid = int((~sample_mask).sum())
    n_times = len(sample_doys)

    rng = np.random.default_rng(42)
    post_bio_tensor = rng.integers(0, 1000, size=(n_valid, 7, n_times)).astype(np.float64)
    post_bio_unc_tensor = rng.random((n_valid, 7, n_times, 7))
    scale_data = rng.random((n_valid, 15)) * 100

    return {
        "post_bio_tensor": post_bio_tensor,
        "post_bio_unc_tensor": post_bio_unc_tensor,
        "scale_data": scale_data,
        "mask": sample_mask,
        "doys": sample_doys,
        "geotransform": sample_geotransform,
        "crs": "EPSG:4326",
    }


@pytest.fixture
def tmp_npz_path(sample_arc_outputs, tmp_path):
    """Save a synthetic NPZ with keys matching ARC's save_data format.

    Keys: post_bio_tensor, post_bio_unc_tensor, dat, geotransform, crs,
          mask, doys, mean_ref, best_candidate
    """
    out = sample_arc_outputs
    rng = np.random.default_rng(99)
    n_valid = out["post_bio_tensor"].shape[0]

    npz_path = tmp_path / "arc_output.npz"
    np.savez(
        npz_path,
        post_bio_tensor=out["post_bio_tensor"],
        post_bio_unc_tensor=out["post_bio_unc_tensor"],
        dat=out["scale_data"],
        geotransform=out["geotransform"],
        crs=np.array(out["crs"]),
        mask=out["mask"],
        doys=out["doys"],
        mean_ref=rng.random((n_valid, 10)),
        best_candidate=rng.integers(0, 5, size=(n_valid,)),
    )
    return npz_path


@pytest.fixture
def sample_weather_ds():
    """xr.Dataset with dims (time,) containing Rin, Rli, Ta, ea, p, u for 6 timesteps."""
    times = pd.date_range("2021-06-01", periods=6, freq="h")
    rng = np.random.default_rng(7)
    return xr.Dataset(
        {
            "Rin": ("time", rng.uniform(200, 800, size=6)),
            "Rli": ("time", rng.uniform(250, 400, size=6)),
            "Ta": ("time", rng.uniform(10, 30, size=6)),
            "ea": ("time", rng.uniform(5, 25, size=6)),
            "p": ("time", rng.uniform(980, 1030, size=6)),
            "u": ("time", rng.uniform(0.5, 10, size=6)),
        },
        coords={"time": times},
    )


@pytest.fixture
def sample_observation_ds():
    """xr.Dataset with solar/viewing angles and delta_time."""
    times = pd.date_range("2021-06-01", periods=6, freq="D")
    rng = np.random.default_rng(11)
    return xr.Dataset(
        {
            "solar_zenith_angle": ("time", rng.uniform(20, 60, size=6)),
            "viewing_zenith_angle": ("time", np.full(6, 0.0)),
            "solar_azimuth_angle": ("time", rng.uniform(100, 260, size=6)),
            "viewing_azimuth_angle": ("time", np.full(6, 0.0)),
            "delta_time": ("time", times.values),
        },
        coords={"time": times},
    )
