import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa
import xarray as xr
from dask_flood_mapper.calculation import (
    bayesian_flood_decision,
    calc_water_likelihood,
    calculate_flood_dc,
    harmonic_expected_backscatter,
)
from dask_flood_mapper.processing import (
    extract_orbit_names,
    post_process_eodc_cube,
    post_process_eodc_cube_,
    post_processing,
    process_datacube,
    process_sig0_dc,
    remove_speckles,
    reproject_equi7grid,
)
from dask_flood_mapper.stac_config import load_config, set_user_config


class MockItemOrbit:
    """Mock class for satellite items."""

    def __init__(self, orbit_state, relative_orbit):
        self.properties = {
            "sat:orbit_state": [orbit_state],
            "sat:relative_orbit": relative_orbit,
        }


temp_dir = tempfile.TemporaryDirectory()
print(temp_dir.name)
USER_CONFIG_DIR = Path(temp_dir.name)


@pytest.fixture
def load_config_test():
    return load_config()


def test_that_config_can_be_loaded(load_config_test):
    assert isinstance(load_config_test, dict)


def test_that_user_config_can_be_set(load_config_test):
    set_user_config(USER_CONFIG_DIR)
    assert load_config() == load_config_test


@pytest.fixture
def mock_items_orbits():
    return [
        MockItemOrbit("ascending", 10),
        MockItemOrbit("descending", 20),
        MockItemOrbit("ascending", 30),
    ]


@pytest.fixture
def mock_item():
    item = MagicMock()
    item.assets = {
        "band1": MagicMock(
            extra_fields={"raster:bands": [{"scale": 2, "nodata": -9999}]}
        )
    }
    return [item]


@pytest.fixture
def dataset():
    return xr.Dataset({"band1": (["x", "y"], [[1, 2], [3, 4]])})


@pytest.fixture
def mock_hpar_dataset():
    # Create a mock dataset with 'time', and the necessary variables for
    # harmonic backscatter computation
    time = pd.date_range(
        "2022-01-01", periods=365, freq="D"
    )  # Create a time range for 1 year
    M0 = np.random.random(365)
    S1 = np.random.random(365)
    S2 = np.random.random(365)
    S3 = np.random.random(365)
    C1 = np.random.random(365)
    C2 = np.random.random(365)
    C3 = np.random.random(365)

    return xr.Dataset(
        {
            "M0": ("time", M0),
            "S1": ("time", S1),
            "S2": ("time", S2),
            "S3": ("time", S3),
            "C1": ("time", C1),
            "C2": ("time", C2),
            "C3": ("time", C3),
        },
        coords={"time": time},
    )


@pytest.fixture
def mock_data_cubes():
    """Mock datasets for merging in flood calculation"""
    coords = {
        "orbit": np.array(["ASCENDING10", "DESCENDING20"]),
        "y": [0, 1],
        "x": [0, 1],
    }

    sig0_dc = xr.Dataset(
        {
            "sig0": (
                ["orbit", "y", "x"],
                da.ones((2, 2, 2), chunks=(1, 2, 2)),
            )
        },
        coords=coords,
    )
    plia_dc = xr.Dataset(
        {
            "plia": (
                ["orbit", "y", "x"],
                da.ones((2, 2, 2), chunks=(1, 2, 2)),
            )
        },
        coords=coords,
    )
    hpar_dc = xr.Dataset(
        {
            "hpar": (
                ["orbit", "y", "x"],
                da.ones((2, 2, 2), chunks=(1, 2, 2)),
            )
        },
        coords=coords,
    )

    return sig0_dc, plia_dc, hpar_dc


@pytest.fixture
def mock_data():
    """Creates a mock dataset similar to real sig0_dc, using Dask arrays."""
    times = np.array(
        ["2022-10-11", "2022-10-11", "2022-10-12"], dtype="datetime64"
    )
    data_values = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
    )

    dask_data = da.from_array(data_values, chunks=(1, 2, 2))

    ds = xr.Dataset(
        {"VV": (["time", "x", "y"], dask_data)},
        coords={"time": times, "x": [0, 1], "y": [0, 1]},
    )

    return ds


class TestExtractOrbitNames:
    def test_extract_orbit_names(self, mock_items_orbits):
        result = extract_orbit_names(mock_items_orbits)
        expected = np.array(["ASCENDING10", "DESCENDING20", "ASCENDING30"])
        assert np.array_equal(result, expected)

    def test_empty_items(self):
        result = extract_orbit_names([])
        expected = np.array([])
        assert np.array_equal(result, expected)

    def test_incomplete_properties(self):
        incomplete_items = [
            MockItemOrbit("ascending", 10),
            MockItemOrbit("", 20),  # Empty orbit state
        ]
        result = extract_orbit_names(incomplete_items)
        expected = np.array(["ASCENDING10", "20"])
        assert np.array_equal(result, expected)


class TestPostProcessEodcCube:
    def test_post_process_eodc_cube_(self, dataset, mock_item):
        expected = dataset / 2
        result = post_process_eodc_cube_(dataset, mock_item, "band1")
        print(f"Dataset:  {dataset.band1}")
        assert result.equals(expected)

    def test_post_process_eodc_cube(self, dataset, mock_item):
        expected = dataset.band1 / 2
        result = post_process_eodc_cube(dataset, mock_item, "band1")

        assert result.band1.equals(expected)


class TestCalculations:
    def test_calc_water_likelihood(self, dataset):
        mplia_values = np.array([[10, 20], [30, 40]])
        dataset["MPLIA"] = (["x", "y"], mplia_values)

        expected_values = mplia_values * -0.394181 + -4.142015

        result = calc_water_likelihood(dataset)

        np.testing.assert_allclose(result.values, expected_values, rtol=1e-5)

    def test_harmonic_expected_backscatter(self, mock_hpar_dataset):
        result = harmonic_expected_backscatter(mock_hpar_dataset)

        w = np.pi * 2 / 365
        t = mock_hpar_dataset.time.dt.dayofyear
        wt = w * t

        M0 = mock_hpar_dataset.M0
        S1 = mock_hpar_dataset.S1
        S2 = mock_hpar_dataset.S2
        S3 = mock_hpar_dataset.S3
        C1 = mock_hpar_dataset.C1
        C2 = mock_hpar_dataset.C2
        C3 = mock_hpar_dataset.C3

        hm_c1 = (M0 + S1 * np.sin(wt)) + (C1 * np.cos(wt))
        hm_c2 = (hm_c1 + S2 * np.sin(2 * wt)) + C2 * np.cos(2 * wt)
        expected = (hm_c2 + S3 * np.sin(3 * wt)) + C3 * np.cos(3 * wt)

        assert (result.values == expected.values).all()

    def test_bayesian_flood_decision(self, mock_hpar_dataset):
        mock_hpar_dataset["sig0"] = (["x", "y"], [[1, 2], [3, 4]])
        mock_hpar_dataset["STD"] = (["x", "y"], [[0.5, 0.5], [0.5, 0.5]])
        mock_hpar_dataset["wbsc"] = (["x", "y"], [[0.1, 0.2], [0.3, 0.4]])
        mock_hpar_dataset["hbsc"] = (["x", "y"], [[0.5, 0.6], [0.7, 0.8]])

        decision = bayesian_flood_decision(mock_hpar_dataset)

        nf_std = 2.754041
        sig0 = mock_hpar_dataset.sig0
        std = mock_hpar_dataset.STD
        wbsc = mock_hpar_dataset.wbsc
        hbsc = mock_hpar_dataset.hbsc

        f_prob = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * (((sig0 - wbsc) / nf_std) ** 2)
        )
        nf_prob = (1.0 / (nf_std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * (((sig0 - hbsc) / nf_std) ** 2)
        )

        evidence = (nf_prob * 0.5) + (f_prob * 0.5)
        nf_post_prob_expected = (nf_prob * 0.5) / evidence
        f_post_prob_expected = (f_prob * 0.5) / evidence
        decision_expected = np.greater(
            f_post_prob_expected, nf_post_prob_expected
        )

        assert (decision.values == decision_expected).all()


class PostProcessing:
    def test_post_processing(self, mock_hpar_dataset):
        mock_hpar_dataset["sig0"] = (["x", "y"], [[10, 15], [30, 35]])
        mock_hpar_dataset["hbsc"] = (["x", "y"], [[0.3, 0.6], [0.7, 0.8]])
        mock_hpar_dataset["wbsc"] = (["x", "y"], [[0.1, 0.4], [0.5, 0.9]])
        mock_hpar_dataset["MPLIA"] = (["x", "y"], [[40, 35], [25, 45]])
        mock_hpar_dataset["STD"] = (["x", "y"], [[1.5, 1.7], [1.6, 1.9]])
        mock_hpar_dataset["f_post_prob"] = (
            ["x", "y"],
            [[0.9, 0.85], [0.75, 0.95]],
        )
        mock_hpar_dataset["decision"] = (
            ["x", "y"],
            [[True, False], [True, True]],
        )

        decision = post_processing(mock_hpar_dataset)

        expected_decision = (
            np.logical_and(
                mock_hpar_dataset.MPLIA >= 27, mock_hpar_dataset.MPLIA <= 48
            )
            * (
                mock_hpar_dataset.hbsc
                > (mock_hpar_dataset.wbsc + 0.5 * 2.754041)
            )
            * (
                (
                    mock_hpar_dataset.sig0
                    > (mock_hpar_dataset.hbsc - 3 * mock_hpar_dataset.STD)
                )
                & (
                    mock_hpar_dataset.sig0
                    < (mock_hpar_dataset.hbsc + 3 * mock_hpar_dataset.STD)
                )
            )
            * (
                mock_hpar_dataset.sig0
                < (mock_hpar_dataset.wbsc + 3 * 2.754041)
            )
            * (mock_hpar_dataset.f_post_prob > 0.8)
        )

        print(f"Resulting decision: {decision.values}")
        print(f"Expected decision: {expected_decision.values}")

        assert np.array_equal(decision.values, expected_decision)


@patch("dask_flood_mapper.processing.post_process_eodc_cube")
@patch("dask_flood_mapper.processing.extract_orbit_names")
def test_process_sig0_dc(
    mock_extract_orbit_names, mock_post_process, mock_data, mock_items_orbits
):
    # Mock function behaviors
    mock_post_process.return_value = mock_data
    mock_extract_orbit_names.return_value = np.array(
        ["ASCENDING10", "DESCENDING20", "ASCENDING30"]
    )

    sig0_dc, orbit_sig0 = process_sig0_dc(mock_data, mock_items_orbits, "VV")

    assert isinstance(
        sig0_dc, xr.Dataset
    ), "Returned object should be an xarray Dataset"
    assert "sig0" in sig0_dc, "Variable should be renamed to 'sig0'"
    assert "orbit" in sig0_dc.coords, "Orbit coordinate should be present"
    assert len(sig0_dc.time) == 2, "Duplicate times should be averaged"

    expected_orbits = np.array(["ASCENDING10", "ASCENDING30"])
    np.testing.assert_array_equal(orbit_sig0, expected_orbits)

    assert sig0_dc.chunks, "Dataset should be persisted (chunked with Dask)"
    assert any(
        isinstance(v.data, da.Array) for v in sig0_dc.data_vars.values()
    ), "Dataset should be persisted (not computed)"


@patch("dask_flood_mapper.processing.post_process_eodc_cube")
@patch("dask_flood_mapper.processing.extract_orbit_names")
def test_process_datacube(
    mock_extract_orbit_names, mock_post_process, mock_data, mock_items_orbits
):
    """Test process_datacube function"""
    mock_post_process.return_value = mock_data
    mock_extract_orbit_names.return_value = np.array(
        ["ASCENDING10", "DESCENDING20", "ASCENDING30"]
    )

    orbit_sig0 = "ASCENDING10"
    bands = "VV"

    result = process_datacube(mock_data, mock_items_orbits, orbit_sig0, bands)

    assert any(
        hasattr(v.data, "chunks") and v.data.chunks is not None
        for v in result.data_vars.values()
    ), "Dataset should be chunked"

    assert result.chunks, "Dataset should be persisted (chunked with Dask)"
    assert any(
        isinstance(v.data, da.Array) for v in result.data_vars.values()
    ), "Dataset should be persisted (not computed)"
    assert any(
        isinstance(v.data, da.Array) for v in result.data_vars.values()
    ), "Dataset should be persisted (not computed)"
    assert any(
        hasattr(v.data, "chunks") and v.data.chunks is not None
        for v in result.data_vars.values()
    ), "Dataset should be chunked"
    assert (
        orbit_sig0 in result.orbit.values
    ), f"Dataset should contain orbit '{orbit_sig0}' only"

    mock_post_process.assert_called_once_with(
        mock_data, mock_items_orbits, bands
    )
    mock_extract_orbit_names.assert_called_once_with(mock_items_orbits)


def test_calculate_flood_dc(mock_data_cubes):
    """Test merging of datasets and flood processing"""
    sig0_dc, plia_dc, hpar_dc = mock_data_cubes
    result = calculate_flood_dc(sig0_dc, plia_dc, hpar_dc)

    assert all(
        var in result.data_vars for var in ["sig0", "plia", "hpar"]
    ), "All variables should be present after merging"

    assert "time" in result.dims, "Orbit dimension should be renamed to 'time'"
    assert "orbit" not in result.dims, "Orbit dimension should be removed"

    assert (
        result.dropna(dim="time", how="all", subset=["sig0"]).sizes["time"]
        == result.sizes["time"]
    ), "All-NaN values in 'sig0' should be removed"

    assert result.chunks, "Dataset should be persisted (chunked with Dask)"
    assert any(
        isinstance(v.data, da.Array) for v in result.data_vars.values()
    ), "Dataset should be persisted (not computed)"


def test_remove_speckles(mock_data_cubes):
    """Test speckle noise removal via rolling median filter"""
    sig0_dc, _, _ = mock_data_cubes  # Use only sig0_dc for filtering
    result = remove_speckles(sig0_dc, window_size=3)

    assert (
        result.sizes["y"] == sig0_dc.sizes["y"]
    ), "y size should remain the same"
    assert (
        result.sizes["x"] == sig0_dc.sizes["x"]
    ), "x size should remain the same"
    assert result.chunks, "Dataset should be persisted (chunked with Dask)"
    assert any(
        isinstance(v.data, da.Array) for v in result.data_vars.values()
    ), "Dataset should be persisted (not computed)"


def assert_datacube_eq(actual, expected):
    xr.testing.assert_allclose(actual, expected, rtol=0.01)


def make_full_datacube(shape, value):
    data = np.full(shape, value).astype(np.float32)
    input_cube = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={
            "y": np.arange(shape[0] * 20, step=20),
            "x": np.arange(shape[1] * 20, step=20),
        },
    ).rio.write_crs("EPSG:27704")
    return input_cube


def make_datacube(values, y, x):
    values_arr = np.array(values).astype(np.float32)
    return xr.DataArray(
        values_arr,
        dims=("latitude", "longitude"),
        coords={"latitude": y, "longitude": x, "spatial_ref": 0},
    )


def test_that_equi7_can_be_reprojected():
    input_cube = make_full_datacube((2, 2), 1)
    output_cube = reproject_equi7grid(
        input_cube,
        [
            -29.983565651546865,
            16.100056529538815,
            -29.982888093209002,
            16.100734087876678,
        ],
    )
    assert_datacube_eq(
        output_cube,
        make_datacube(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, 1.0, np.nan, np.nan],
                [np.nan, 1.0, 1.0, 1.0, np.nan],
                [np.nan, np.nan, 1.0, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            x=[-29.9, -29.9, -29.9, -29.9, -29.9],
            y=[16.1, 16.1, 16.1, 16.1, 16.1],
        ),
    )


if __name__ == "__main__":
    pytest.main()
