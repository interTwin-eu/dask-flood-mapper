import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask_flood_mapper.harmonic_params import (
    harmonic_regression,
    model_coords,
    process_harmonic_parameters_datacube,
    reduce_to_harmonic_parameters,
)


def generate_harmonic_timeseries(times, mean, sin_amplitudes, cos_amplitudes):
    """Generate a time series from harmonic parameters."""
    w = 2 * np.pi / 365
    result = mean * np.ones_like(times)

    for k, (sin_amp, cos_amp) in enumerate(
        zip(sin_amplitudes, cos_amplitudes), 1
    ):
        result += sin_amp * np.sin(k * w * times) + cos_amp * np.cos(
            k * w * times
        )

    return result


@pytest.fixture(
    params=[
        # test some different Ks
        {"mean": 0.0, "sin_amplitudes": [0.5], "cos_amplitudes": [0.5]},
        {
            "mean": 0.5,
            "sin_amplitudes": [0.3, 0.1],
            "cos_amplitudes": [0.2, 0.05],
        },
        {
            "mean": -0.5,
            "sin_amplitudes": [0.6, 0.2, 0.1],
            "cos_amplitudes": [0.3, 0.1, 0.2],
        },
        # typical case
        {
            "mean": -10,
            "sin_amplitudes": [-0.1, 0.05, 0.06],
            "cos_amplitudes": [0.1, 2, -0.3],
        },
        # test values at edges of realistic range
        {
            "mean": -30,
            "sin_amplitudes": [-5, -4, -1.5],
            "cos_amplitudes": [-9, -2, -1.5],
        },
        {
            "mean": 5,
            "sin_amplitudes": [5, 4, 1.5],
            "cos_amplitudes": [9, 4, 1.1],
        },
        {
            "mean": 5,
            "sin_amplitudes": [-5, -4, -1.5],
            "cos_amplitudes": [-9, -2, -1.5],
        },
        {
            "mean": -30,
            "sin_amplitudes": [5, 4, 1.5],
            "cos_amplitudes": [9, 4, 1.1],
        },
        {
            "mean": -30,
            "sin_amplitudes": [5, 4, 1.5],
            "cos_amplitudes": [-9, -2, -1.5],
        },
        {
            "mean": 5,
            "sin_amplitudes": [-5, -4, -1.5],
            "cos_amplitudes": [9, 4, 1.1],
        },
        {
            "mean": -30,
            "sin_amplitudes": [-5, -4, -1.5],
            "cos_amplitudes": [9, 4, 1.1],
        },
        {
            "mean": 5,
            "sin_amplitudes": [5, 4, 1.5],
            "cos_amplitudes": [-9, -2, -1.5],
        },
        {
            "mean": -30,
            "sin_amplitudes": [-5, 4, 1.5],
            "cos_amplitudes": [9, 4, -1.5],
        },
        {
            "mean": 5,
            "sin_amplitudes": [5, -4, -1.5],
            "cos_amplitudes": [-9, -2, 1.1],
        },
    ]
)
def synthetic_data(request):
    # Extract parameters
    mean = request.param["mean"]
    sin_amplitudes = request.param["sin_amplitudes"]
    cos_amplitudes = request.param["cos_amplitudes"]

    # Create time series
    times = np.sort(np.random.randint(1, 366, size=400)).astype(
        np.float32
    )  # Random day of the year
    # times = np.sort(np.hstack([times, times + 1]))
    rows, cols = 2, 2

    orbit = np.array([["A1", "B1"][int(time % 2)] for time in times])
    # Generate perfect harmonic signal
    ts = generate_harmonic_timeseries(
        times, mean, sin_amplitudes, cos_amplitudes
    )
    ts_data = ts.reshape(-1, 1, 1).astype(np.float32)
    ts_data = np.broadcast_to(ts_data, (len(times), rows, cols)).copy()

    expected_params = np.array(
        [mean]
        + [val for pair in zip(sin_amplitudes, cos_amplitudes) for val in pair]
        + [0.0, len(times)]
    )  # std=0 since no noise, nobs=len(times)

    return {
        "times": times,
        "data": ts_data,
        "orbit": orbit,
        "expected_params": expected_params,
        "k": len(sin_amplitudes),
    }


def test_harmonic_regression_returns_accurate_params(synthetic_data):
    # Run harmonic regression
    params = harmonic_regression(
        synthetic_data["data"], synthetic_data["times"], k=synthetic_data["k"]
    )

    # Check output shape
    # mean + 2*harmonics + std + nobs
    expected_n_params = 2 * synthetic_data["k"] + 1 + 2
    assert params.shape == (expected_n_params, 2, 2)

    # Check if parameters match expected values (within numerical precision)
    for i in range(2):
        for j in range(2):
            np.testing.assert_allclose(
                params[:, i, j],
                synthetic_data["expected_params"],
                rtol=1e-4,
                atol=1e-4,
            )


def test_harmonic_regression_handles_nan_values(synthetic_data):
    # Insert some NaN values into the data
    data_with_nans = synthetic_data["data"].copy()
    # Make 2 observations NaN in first pixel
    data_with_nans[3:5, 0, 0] = np.nan

    # Run harmonic regression
    params = harmonic_regression(
        data_with_nans, synthetic_data["times"], k=synthetic_data["k"]
    )

    # Check that parameters are still reasonable
    assert not np.isnan(params[:, 0, 0]).any(), "Parameters should not be NaN"
    assert (
        params[-1, 0, 0] == len(synthetic_data["times"]) - 2
    ), "nobs should reflect missing values"
    assert params[-1, 0, 1] == len(
        synthetic_data["times"]
    ), "Other pixels should have full observations"


def test_harmonic_regression_handles_insufficient_data(synthetic_data):
    # Make most data NaN to trigger insufficient data condition
    data_with_many_nans = synthetic_data["data"].copy()
    synthetic_k = synthetic_data["k"]
    data_with_many_nans[
        : (synthetic_data["data"].shape[0] - 2 * synthetic_k), 0, 0
    ] = np.nan  # Leave only 2*k observations

    # Run harmonic regression which requires at least 2k+1 observations
    params = harmonic_regression(
        data_with_many_nans, synthetic_data["times"], k=synthetic_k
    )

    assert np.isnan(
        params[:-1, 0, 0]
    ).all(), "Parameters should be NaN with insufficient data"
    assert (
        params[-1, 0, 0] == 2 * synthetic_k
    ), f"NOBS should be {2 * synthetic_k}"
    assert not np.isnan(
        params[:, 0, 1]
    ).any(), "Other pixels should have valid parameters"


def test_harmonic_regression_respects_redundancy(synthetic_data):
    # Make some data NaN but keep enough for default redundancy
    data_with_nans = synthetic_data["data"].copy()
    k = synthetic_data["k"]
    data_with_nans[: data_with_nans.shape[0] - (2 * k + 2), 0, 0] = (
        np.nan
    )  # Leave 6 observations

    # Should work with redundancy=1
    params_red1 = harmonic_regression(
        data_with_nans,
        synthetic_data["times"],
        k=synthetic_data["k"],
        redundancy=1,
    )
    assert not np.isnan(
        params_red1[:-1, 0, 0]
    ).any(), "Should work with redundancy=1"

    # Should fail with redundancy=2
    params_red2 = harmonic_regression(
        data_with_nans,
        synthetic_data["times"],
        k=synthetic_data["k"],
        redundancy=2,
    )
    assert np.isnan(
        params_red2[:-1, 0, 0]
    ).all(), "Should fail with redundancy=2"


def test_harmonic_regression_handles_no_data(synthetic_data):
    # Make all data NaN to trigger insufficient data condition
    data_with_many_nans = synthetic_data["data"].copy()
    data_with_many_nans[:, 0, 0] = np.nan  # Leave only 4 observations

    # Run harmonic regression with k=2 (requires at least 5 observations)
    params = harmonic_regression(
        data_with_many_nans, synthetic_data["times"], k=synthetic_data["k"]
    )

    assert np.isnan(
        params[:, 0, 0]
    ).all(), "Parameters including NOBS should be NaN with no data"
    assert not np.isnan(
        params[:, 0, 1]
    ).any(), "Other pixels should have valid parameters"


@pytest.fixture
def synthetic_xarray_data(synthetic_data):
    # Create synthetic xarray DataArray
    times = synthetic_data["times"]
    data = synthetic_data["data"]

    return xr.DataArray(
        # jscpd:ignore-start
        data=data,
        coords={
            "time": times,
            "y": np.arange(data.shape[1]),
            "x": np.arange(data.shape[2]),
        },
        dims=["time", "y", "x"],
    )
    # jscpd:ignore-end


@pytest.fixture
def synthetic_xarray_dataset(synthetic_data):
    # Create synthetic xarray DataArray
    times = pd.to_timedelta(synthetic_data["times"], "D") + np.datetime64(
        "2018-12-31"
    )
    data = synthetic_data["data"]
    orbit = synthetic_data["orbit"]

    sig_da = xr.DataArray(
        name="sig0",
        data=data,
        coords={
            "time": times,
            "y": np.arange(data.shape[1]),
            "x": np.arange(data.shape[2]),
        },
        dims=["time", "y", "x"],
    )
    orbit_da = xr.DataArray(
        name="orbit",
        data=orbit,
        coords={
            "time": times,
        },
        dims=["time"],
    )
    return xr.merge([sig_da, orbit_da])


def test_synthetic_array_dataset_contains_original_synthetic_data(
    synthetic_xarray_dataset, synthetic_data
):
    # Check that the synthetic dataset contains the original synthetic data
    np.testing.assert_array_equal(
        synthetic_xarray_dataset["sig0"].values, synthetic_data["data"]
    ), "Synthetic dataset does not contain the original synthetic data"
    np.testing.assert_array_equal(
        synthetic_xarray_dataset["orbit"].values, synthetic_data["orbit"]
    ), "Synthetic dataset does not contain the original orbit data"
    np.testing.assert_array_equal(
        synthetic_xarray_dataset["time.dayofyear"].values,
        synthetic_data["times"],
    ), "Synthetic dataset does not contain the original time data"


def test_reduce_to_harmonic_parameters_basic(
    synthetic_xarray_data, synthetic_data
):
    # Run reduction
    result = reduce_to_harmonic_parameters(
        synthetic_xarray_data,
        dtimes=synthetic_data["times"],
        k=synthetic_data["k"],
    )

    # Check basic properties
    assert isinstance(result, xr.DataArray)
    assert set(result.dims) == {"param", "y", "x"}
    assert result.shape[1:] == synthetic_xarray_data.shape[1:]

    # Check parameter values match direct regression
    direct_params = harmonic_regression(
        synthetic_xarray_data.values,
        synthetic_data["times"],
        k=synthetic_data["k"],
    )
    np.testing.assert_allclose(result.values, direct_params)


def test_reduce_to_harmonic_parameters_coordinates(synthetic_xarray_data):
    k = 2
    result = reduce_to_harmonic_parameters(
        synthetic_xarray_data, dtimes=synthetic_xarray_data.time.values, k=k
    )

    # Check coordinates are properly set
    expected_params = model_coords(k)
    assert list(result.param.values) == expected_params
    np.testing.assert_array_equal(
        result.x.values, synthetic_xarray_data.x.values
    )
    np.testing.assert_array_equal(
        result.y.values, synthetic_xarray_data.y.values
    )


def test_reduce_to_harmonic_parameters_with_nans(synthetic_xarray_data):
    # Add some NaN values
    data_with_nans = synthetic_xarray_data.copy()
    data_with_nans[3:5, 0, 0] = np.nan

    result = reduce_to_harmonic_parameters(
        data_with_nans, dtimes=synthetic_xarray_data.time.values, k=2
    )

    # Check that parameters are computed correctly despite NaNs
    assert not np.isnan(result.sel(x=0, y=0)).all()
    assert (
        result.sel(param="NOBS", x=0, y=0)
        == len(synthetic_xarray_data.time) - 2
    )


@pytest.fixture
def make_pars_list(synthetic_xarray_dataset, synthetic_data):
    harm_pars_list = []
    for orbit, orbit_ds in synthetic_xarray_dataset.groupby("orbit"):
        dtimes = orbit_ds["time.dayofyear"]
        harm_pars = reduce_to_harmonic_parameters(
            orbit_ds["sig0"], dtimes=dtimes, k=synthetic_data["k"]
        )
        harm_pars_list.append((orbit, harm_pars))
    return harm_pars_list


def assert_both_orbits_have_approx_the_same_parameters(hpar_dc):
    assert np.all(
        np.abs(hpar_dc.diff(dim="orbit")) < 1e-6
    ), "Orbits differ too much"


def assert_retrieved_harmpars_are_approx_the_same_as_synthetic_data(
    synthetic_data, hpar_dc
):
    retrieved_params = hpar_dc.to_dataarray(dim="param")
    expected_params = xr.DataArray(
        data=synthetic_data["expected_params"],
        dims=["param"],
        coords={"param": model_coords(synthetic_data["k"])},
    )
    diffs = retrieved_params - expected_params
    assert np.all(
        np.abs(diffs) < 1e-4
    ), f"Retrieved parameters differ from expected: {diffs}"


def assert_sig0_only_has_valid_times(sig0_dc, time_range):
    assert (
        sig0_dc.time.min() >= time_range[0]
    ), "sig0_dc contains times before the start of the range"
    assert (
        sig0_dc.time.max() <= time_range[1]
    ), "sig0_dc contains times after the end of the range"


def assert_we_have_hpars_for_all_sig0_orbits(sig0_dc, hpar_dc, orbit_sig0):
    sig0_orbits = sig0_dc.orbit.values
    assert np.all(
        np.isin(sig0_orbits, hpar_dc.orbit.values)
    ), "Not all sig0 orbits have harmonic parameters"


def test_make_process(
    make_pars_list, synthetic_data, synthetic_xarray_dataset
):
    pars_list = make_pars_list.copy()
    sig0_dc = synthetic_xarray_dataset.copy().sortby("time")
    time_range = (sig0_dc.time[-4].data, sig0_dc.time[-1].data)
    sig0_dc, hpar_dc, orbit_sig0 = process_harmonic_parameters_datacube(
        sig0_dc, time_range, pars_list, min_nobs=32
    )
    assert_both_orbits_have_approx_the_same_parameters(hpar_dc)
    assert_retrieved_harmpars_are_approx_the_same_as_synthetic_data(
        synthetic_data, hpar_dc
    )
    assert_sig0_only_has_valid_times(sig0_dc, time_range)
    assert_we_have_hpars_for_all_sig0_orbits(sig0_dc, hpar_dc, orbit_sig0)
    np.testing.assert_array_equal(orbit_sig0, hpar_dc.orbit)
