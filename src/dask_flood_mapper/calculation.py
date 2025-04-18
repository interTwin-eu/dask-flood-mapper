import numpy as np
import xarray as xr


def calculate_flood_dc(sig0_dc, plia_dc, hpar_dc):
    """
    Merge four data cubes and apply processing steps to clean and filter the dataset.
    wcover_dc is optional.
    """
    flood_dc = xr.merge([sig0_dc, plia_dc, hpar_dc])
    flood_dc = (
        flood_dc.reset_index("orbit", drop=True)
        .rename({"orbit": "time"})
        .dropna(dim="time", how="all", subset=["sig0"])
    )
    flood_dc = flood_dc.persist()
    return flood_dc


def remove_speckles(flood_output, window_size=5):
    """
    Apply a rolling median filter to smooth the dataset spatially over longitude
    and latitude.
    """
    flood_output = (
        flood_output.rolling({"x": window_size, "y": window_size}, center=True)
        .median(skipna=True)
        .persist()
    )
    return flood_output


def calc_water_likelihood(dc):
    return dc.MPLIA * -0.394181 + -4.142015


def harmonic_expected_backscatter(dc):
    w = np.pi * 2 / 365

    t = dc.time.dt.dayofyear
    wt = w * t

    M0 = dc.M0
    S1 = dc.S1
    S2 = dc.S2
    S3 = dc.S3
    C1 = dc.C1
    C2 = dc.C2
    C3 = dc.C3
    hm_c1 = (M0 + S1 * np.sin(wt)) + (C1 * np.cos(wt))
    hm_c2 = (hm_c1 + S2 * np.sin(2 * wt)) + C2 * np.cos(2 * wt)
    hm_c3 = (hm_c2 + S3 * np.sin(3 * wt)) + C3 * np.cos(3 * wt)
    return hm_c3


def bayesian_flood_decision(dc):
    f_prob, nf_prob = calc_prior_probability(dc)
    decision = xr.where(
        np.isnan(f_prob) | np.isnan(nf_prob),
        np.nan,
        np.greater(f_prob, nf_prob),
    )
    return decision


def bayesian_flood_probability(dc):
    f_prob, nf_prob = calc_prior_probability(dc)
    evidence = (nf_prob * 0.5) + (f_prob * 0.5)
    return (f_prob * 0.5) / evidence


def calc_prior_probability(dc):
    nf_std = 2.754041
    sig0 = dc.sig0
    std = dc.STD
    wbsc = dc.wbsc
    hbsc = dc.hbsc
    f_prob = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * (((sig0 - wbsc) / nf_std) ** 2)
    )
    nf_prob = (1.0 / (nf_std * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * (((sig0 - hbsc) / nf_std) ** 2)
    )
    return f_prob, nf_prob
