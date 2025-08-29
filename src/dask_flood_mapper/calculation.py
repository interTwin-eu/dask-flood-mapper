"""Functions to calculate flood-related metrics."""

import numpy as np
import xarray as xr


def calculate_flood_dc(
    sig0_dc: xr.DataArray | xr.Dataset,
    plia_dc: xr.DataArray | xr.Dataset,
    hpar_dc: xr.DataArray | xr.Dataset,
) -> xr.Dataset:
    """Merge three data cubes.

    Merge three data cubes and apply processing steps to clean
    and filter the dataset. wcover_dc is optional.
    """
    # TODO: Add wcover_dc to the calculation  # noqa: FIX002
    flood_dc: xr.Dataset = xr.merge([sig0_dc, plia_dc, hpar_dc])
    flood_dc: xr.Dataset = (  # type: ignore
        flood_dc.reset_index("orbit", drop=True)
        .rename({"orbit": "time"})
        .dropna(dim="time", how="all", subset=["sig0"])
    )
    return flood_dc.persist()


def calc_water_likelihood(dc: xr.Dataset) -> xr.DataArray:
    """Calculate the water likelihood based on the MPLIA value."""
    return dc.MPLIA * -0.394181 + -4.142015


def harmonic_expected_backscatter(dc: xr.Dataset) -> xr.DataArray:
    """Calculate the harmonic expected backscatter."""
    w: float = np.pi * 2 / 365

    t: xr.DataArray = dc.time.dt.dayofyear
    wt: xr.DataArray = w * t
    wt2: xr.DataArray = wt * 2
    wt3: xr.DataArray = wt * 3

    hm_c1: xr.DataArray = (dc.M0 + dc.S1 * np.sin(wt)) + (dc.C1 * np.cos(wt))
    hm_c2: xr.DataArray = (hm_c1 + dc.S2 * np.sin(wt2)) + dc.C2 * np.cos(wt2)
    return (hm_c2 + dc.S3 * np.sin(wt3)) + dc.C3 * np.cos(wt3)


def bayesian_flood_decision(dc: xr.Dataset) -> xr.DataArray:
    """Make a flood decision based on the prior probabilities."""
    f_prob, nf_prob = calc_prior_probability(dc)
    return xr.where(
        np.isnan(f_prob) | np.isnan(nf_prob),
        np.nan,
        np.greater(f_prob, nf_prob),
    )


def bayesian_flood_probability(dc: xr.Dataset) -> xr.DataArray:
    """Calculate the baysian flood probability."""
    flood_prob, nonflood_prob = calc_prior_probability(dc)
    evidence: xr.DataArray = (nonflood_prob * 0.5) + (flood_prob * 0.5)
    return (flood_prob * 0.5) / evidence


def calc_prior_probability(
    dc: xr.Dataset,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Calculate the prior probabilities for flood and non-flood."""
    nf_std: float = 2.754041
    sig0: xr.DataArray = dc.sig0
    std: xr.DataArray = dc.STD
    wbsc: xr.DataArray = dc.wbsc
    hbsc: xr.DataArray = dc.hbsc
    flood_prob: xr.DataArray = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * (((sig0 - wbsc) / nf_std) ** 2),
    )
    nonflood_prob: xr.DataArray = (
        1.0 / (nf_std * np.sqrt(2 * np.pi))
    ) * np.exp(
        -0.5 * (((sig0 - hbsc) / nf_std) ** 2),
    )
    return flood_prob, nonflood_prob
