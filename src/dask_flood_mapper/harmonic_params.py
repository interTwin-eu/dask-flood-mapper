"""Compute harmonic parameters for a time series of SAR data."""

import datetime as dt

import numpy as np
import xarray as xr
from numba import njit, prange

from dask_flood_mapper.processing import order_orbits


def create_harmonic_parameters(
    sig0_dc: xr.Dataset,
) -> list[tuple[int, xr.DataArray]]:
    """Create harmonic parameters for each orbit in the sig0 datacube."""
    harm_pars_list: list = []
    for orbit, orbit_ds in sig0_dc.groupby("orbit"):
        orbit_ds: xr.Dataset = orbit_ds.chunk({"time": -1}).persist()  # noqa
        dtimes: xr.DataArray = orbit_ds["time.dayofyear"].compute()
        harm_pars: xr.DataArray = xr.map_blocks(
            func=reduce_to_harmonic_parameters,
            obj=orbit_ds["sig0"],
            kwargs={
                "dtimes": dtimes,
                "k": 3,
                "x_var_name": "x",
                "y_var_name": "y",
            },
        ).persist()
        harm_pars_list.append((orbit, harm_pars))
    return harm_pars_list


def process_harmonic_parameters_datacube(
    sig0_dc: xr.Dataset,  # type: ignore
    time_range: tuple[dt.datetime, ...],
    harm_pars_list: list[tuple[int, xr.DataArray]],
    min_nobs: int = 32,
) -> tuple[xr.Dataset, xr.Dataset, np.ndarray]:
    """Process the harmonic parameters datacube."""
    hpar_dc: xr.DataArray = xr.concat(  # type: ignore
        [harm_pars[1] for harm_pars in harm_pars_list],
        dim="orbit",
    )
    hpar_dc: xr.DataArray = hpar_dc.where(  # type: ignore
        hpar_dc.sel(param="NOBS") >= min_nobs,
    ).drop_sel(
        param="NOBS",
    )
    hpar_dc: xr.Dataset = hpar_dc.to_dataset(dim="param")
    hpar_dc: xr.Dataset = hpar_dc.assign_coords(
        orbit=np.array([harm_pars[0] for harm_pars in harm_pars_list]),
    )

    # time range of flood map
    if len(time_range) == 1:
        sig0_dc: xr.Dataset = sig0_dc.sel(time=time_range, method="nearest")
    else:
        sig0_dc: xr.Dataset = sig0_dc.sel(
            time=slice(time_range[0], time_range[1]),
        )
    orbit_sig0: np.ndarray = order_orbits(sig0_dc)
    hpar_dc: xr.Dataset = hpar_dc.sel(orbit=orbit_sig0)
    hpar_dc: xr.Dataset = hpar_dc.persist()
    return sig0_dc, hpar_dc, orbit_sig0


def reduce_to_harmonic_parameters(
    ts_xr: xr.DataArray,
    dtimes: np.ndarray,
    x_var_name: str = "x",
    y_var_name: str = "y",
    **kwargs,  # noqa: ANN003
) -> xr.DataArray:
    """Reduce a time series to harmonic parameters."""
    params_arr = harmonic_regression(ts_xr.data, dtimes=dtimes, **kwargs)
    k: int = kwargs.get("k", 3)
    out_dims: list[str] = ["param", y_var_name, x_var_name]
    return xr.DataArray(
        data=params_arr,
        coords={
            "param": model_coords(k),
            x_var_name: ts_xr[x_var_name],
            y_var_name: ts_xr[y_var_name],
        },
        dims=out_dims,
    )


def harmonic_regression(
    arr: np.ndarray,
    dtimes: np.ndarray,
    k: int = 3,
    redundancy: int = 1,
) -> np.ndarray:
    """Perform harmonic regression on an Array."""
    # define constants
    w: float = np.pi * 2 / 365

    # should be in dayofyear format
    t = dtimes

    # prepare A-matrix
    num_dims: int = 3
    if len(arr.shape) != num_dims:
        msg: str = "Input array must be 3D (time, rows, cols)."
        raise ValueError(msg)

    ti, rows, cols = arr.shape
    nx: int = 2 * k + 1
    a = [np.ones_like(t)]
    for i in range(1, k + 1):
        a += [np.sin(i * w * t), np.cos(i * w * t)]
    a = np.vstack(a).T.astype(np.float32)

    # run regression
    param = np.full((nx + 2, rows, cols), np.nan, dtype=np.float32)
    _fast_harmonic_regression(
        arr=arr,
        a_matrix=a,
        k=k,
        red=redundancy,
        param=param,
    )
    return param


@njit(parallel=True)
def _fast_harmonic_regression(
    arr: np.ndarray,
    a_matrix: np.ndarray,
    red: int,
    param: np.ndarray,
    k: int = 3,
) -> None:
    # loop through rows and columns
    ti, rows, cols = arr.shape
    nx = a_matrix.shape[1]
    for row in prange(rows):  # type: ignore
        for col in prange(cols):  # type: ignore
            # remove NaN values
            l_unfiltered = arr[:, row, col]
            valid_obs = ~np.isnan(l_unfiltered)
            A, l = a_matrix[valid_obs, :], l_unfiltered[valid_obs]  # noqa

            # N should be nan if no observations,
            # otherwise sum of valid observations
            # even if there aren't enough to calculate a good solution
            N = np.sum(valid_obs)  # noqa: N806
            param[-1, row, col] = N or np.nan

            if (red * nx) <= l.shape[0]:
                # calculate least-squares solution, residuals and valid
                # observations
                px_x = np.linalg.lstsq(A, l)[0]
                v = np.dot(A, px_x) - l

                # calculate standard deviation using SSE
                denom = N - (2 * k + 1)
                if denom == 0:
                    px_std = np.nan
                else:
                    px_std = np.sqrt(np.sum(v**2) / (N - (2 * k + 1)))

                # add pixel result to return array
                param[:-2, row, col] = px_x
                param[-2, row, col] = px_std


def model_coords(kvalue: int) -> list[str]:
    """Create a list of model coordinates for harmonic parameters."""
    coord_list: list[str] = ["M0"]
    for n in range(1, kvalue + 1):
        coord_list.extend(["S" + str(n), "C" + str(n)])
    coord_list.append("STD")
    coord_list.append("NOBS")
    return coord_list
