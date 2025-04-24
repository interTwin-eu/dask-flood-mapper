import numpy as np
import rioxarray  # noqa
import xarray as xr
import dask.array as da
from dask_flood_mapper.catalog import config
from odc import stac as odc_stac
from odc.geo.xr import ODCExtensionDa
from numba import njit, prange

# import parameters from config.yaml file
crs = config["base"]["crs"]
chunks = config["base"]["chunks"]
groupby = config["base"]["groupby"]
BANDS_HPAR = (
    "C1",
    "C2",
    "C3",
    "M0",
    "S1",
    "S2",
    "S3",
    "STD",
)  # not possible to add to yaml file since is a ("a", "v") type
BANDS_PLIA = "MPLIA"


# pre-processing
def prepare_dc(items, bbox, bands):
    return odc_stac.load(
        items,
        bands=bands,
        chunks=chunks,
        bbox=bbox,
        groupby=groupby,
    )


# processing
def process_sig0_dc(sig0_dc, items_sig0, bands):
    sig0_dc = (
        post_process_eodc_cube(sig0_dc, items_sig0, bands)
        .rename_vars({"VV": "sig0"})
        .assign_coords(orbit=("time", extract_orbit_names(items_sig0)))
        .dropna(dim="time", how="all")
        .sortby("time")
    )
    __, indices = np.unique(sig0_dc.time, return_index=True)
    indices.sort()
    orbit_sig0 = sig0_dc.orbit[indices].data
    sig0_dc = sig0_dc.groupby("time").mean(skipna=True)
    sig0_dc = sig0_dc.assign_coords(orbit=("time", orbit_sig0))
    sig0_dc = sig0_dc.persist()
    return sig0_dc, orbit_sig0


def process_datacube(datacube, items_dc, orbit_sig0, bands):
    datacube = post_process_eodc_cube(datacube, items_dc, bands).rename(
        {"time": "orbit"}
    )
    datacube["orbit"] = extract_orbit_names(items_dc)
    datacube = datacube.groupby("orbit").mean(skipna=True)
    datacube = datacube.sel(orbit=orbit_sig0)
    datacube = datacube.persist()
    return datacube


# post-processing
def post_process_eodc_cube(dc: xr.Dataset, items, bands):
    if not isinstance(bands, tuple):
        bands = tuple([bands])
    for i in bands:
        dc[i] = post_process_eodc_cube_(dc[i], items, i)
    return dc


def post_process_eodc_cube_(dc: xr.DataArray, items, band):
    scale = items[0].assets[band].extra_fields.get("raster:bands")[0]["scale"]
    nodata = items[0].assets[band].extra_fields.get("raster:bands")[0]["nodata"]
    # Apply the scaling and nodata masking logic
    return dc.where(dc != nodata) / scale


def extract_orbit_names(items):
    return np.array(
        [
            items[i].properties["sat:orbit_state"][0].upper()
            + str(items[i].properties["sat:relative_orbit"])
            for i in range(len(items))
        ]
    )


def post_processing(dc):
    dc = dc * np.logical_and(dc.MPLIA >= 27, dc.MPLIA <= 48)
    dc = dc * (dc.hbsc > (dc.wbsc + 0.5 * 2.754041))
    land_bsc_lower = dc.hbsc - 3 * dc.STD
    land_bsc_upper = dc.hbsc + 3 * dc.STD
    water_bsc_upper = dc.wbsc + 3 * 2.754041
    mask_land_outliers = np.logical_and(
        dc.sig0 > land_bsc_lower, dc.sig0 < land_bsc_upper
    )
    mask_water_outliers = dc.sig0 < water_bsc_upper
    dc = dc * (mask_land_outliers | mask_water_outliers)
    return (dc * (dc.f_post_prob > 0.8)).decision


def reproject_equi7grid(dc, bbox, target_epsg=crs):
    return ODCExtensionDa(dc).reproject(target_epsg).rio.clip_box(*bbox)


def reduce_to_harmonic_parameters(
        ts_xr: xr.DataArray,
        x_var_name = 'x',
        y_var_name = 'y',
        **kwargs
):
    params_arr = harmonic_regression(ts_xr.data, **kwargs)
    k = kwargs.get('k', 3)
    out_dims = ['param', y_var_name, x_var_name]
    out_dataarray = xr.DataArray(
        data=params_arr,
        coords={
            'param': model_coords(k),
            x_var_name: ts_xr[x_var_name],
            y_var_name: ts_xr[y_var_name]
        },
        dims=out_dims
    )
    return out_dataarray


def harmonic_regression(
        arr: np.ndarray,
        dtimes: np.ndarray,
        k: int = 3,
        redundancy: int = 1,
        axis: int = 0
) -> np.ndarray:
    # define constants
    w = np.pi * 2 / 365

    # should be in dayofyear format
    t = dtimes

    # prepare A-matrix
    ti, rows, cols = arr.shape
    nx = 2 * k + 1
    a = [np.ones_like(t)]
    for i in range(1, k + 1):
        a += [np.sin(i * w * t), np.cos(i * w * t)]
    a = np.vstack(a).T.astype(np.float32)

    # run regression
    param = np.full((nx + 2, rows, cols), np.nan, dtype=np.float32)
    _fast_harmonic_regression(arr=arr, a_matrix=a, k=k,
                              red=redundancy, param=param)

    return param


@njit(parallel=True)
def _fast_harmonic_regression(arr, a_matrix, red, param, k=3):
    # loop through rows and columns
    ti, rows, cols = arr.shape
    nx = a_matrix.shape[1]
    for row in prange(rows):
        for col in prange(cols):
            # remove NaN values
            l_unfiltered = arr[:, row, col]
            valid_obs = ~np.isnan(l_unfiltered)
            A, l = a_matrix[valid_obs, :], l_unfiltered[valid_obs]

            # N should be nan if no observations, otherwise sum of valid observations
            # even if there aren't enough to calculate a good solution
            N = np.sum(valid_obs)
            param[-1, row, col] = N or np.nan

            if (red * nx) <= l.shape[0]:
                # calculate least-squares solution, residuals and valid observations
                px_x = np.linalg.lstsq(A, l)[0]
                v = np.dot(A, px_x) - l

                # calculate standard deviation using SSE
                denom = N - (2 * k + 1)
                if denom == 0:
                    px_std = np.nan
                else:
                    px_std = np.sqrt(np.sum(v ** 2) / (N - (2 * k + 1)))

                # add pixel result to return array
                param[:-2, row, col] = px_x
                param[-2, row, col] = px_std


def model_coords(kvalue):
    coord_list = ["M0"]
    for n in range(1, kvalue + 1):
        coord_list.extend(["S" + str(n), "C" + str(n)])
    coord_list.append("STD")
    coord_list.append("NOBS")
    return coord_list