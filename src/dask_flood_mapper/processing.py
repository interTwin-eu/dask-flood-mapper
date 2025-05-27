import numpy as np
import rioxarray  # noqa
import xarray as xr
from dask_flood_mapper.catalog import config
from odc import stac as odc_stac
from odc.geo.xr import ODCExtensionDa

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
    orbit_sig0 = order_orbits(sig0_dc)
    sig0_dc = sig0_dc.groupby("time").mean(skipna=True)
    sig0_dc = sig0_dc.assign_coords(orbit=("time", orbit_sig0))
    sig0_dc = sig0_dc.persist()
    return sig0_dc, orbit_sig0


def order_orbits(sig0_dc):
    if sig0_dc.time.shape != ():
        __, indices = np.unique(sig0_dc.time, return_index=True)
        indices.sort()
        return sig0_dc.orbit[indices].data
    else:
        return np.array([sig0_dc.orbit.data])


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
    dc["mask_exceeding_PLIA"] = mask_exceeding_PLIA(dc)
    dc["mask_conflicting_distributions"] = mask_conflicting_distributions(dc)
    dc["mask_outliers"] = mask_outliers(dc)
    dc["mask_denial_high_uncertainty"] = mask_denial_high_uncertainty(dc)
    dc["decision"] = (
        dc.decision
        * dc["mask_exceeding_PLIA"]
        * dc["mask_conflicting_distributions"]
        * dc["mask_outliers"]
        * dc["mask_denial_high_uncertainty"]
    )
    dc["decision"] = remove_speckles(dc.decision)
    return dc.decision


def mask_denial_high_uncertainty(dc):
    return np.minimum(dc.nf_post_prob, dc.f_post_prob) > 0.2


def mask_conflicting_distributions(dc):
    return dc.hbsc > (dc.wbsc + 0.5 * 2.754041)


def mask_outliers(dc):
    land_bsc_lower = dc.hbsc - 3 * dc.STD
    land_bsc_upper = dc.hbsc + 3 * dc.STD
    water_bsc_upper = dc.wbsc + 3 * 2.754041
    mask_land_outliers = np.logical_and(
        dc.sig0 > land_bsc_lower, dc.sig0 < land_bsc_upper
    )
    mask_water_outliers = dc.sig0 < water_bsc_upper
    return mask_land_outliers | mask_water_outliers


def mask_exceeding_PLIA(dc):
    return np.logical_and(dc.MPLIA >= 27, dc.MPLIA <= 48)


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


def reproject_equi7grid(dc, bbox, target_epsg=crs):
    return ODCExtensionDa(dc).reproject(target_epsg).rio.clip_box(*bbox)
