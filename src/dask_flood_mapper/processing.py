"""Processing, and post-processing functions for Dask Flood Mapper."""

from typing import Literal

import numpy as np
import rioxarray  # noqa
import xarray as xr
from odc import stac as odc_stac
from odc.geo.xr import ODCExtensionDa
from pystac.item_collection import ItemCollection

from dask_flood_mapper.catalog import config

# import parameters from config.yaml file
CRS: str = config["base"]["crs"]
CHUNKS: dict[str, int | Literal["auto"]] | None = config["base"]["chunks"]
GROUPBY: str | None = config["base"]["groupby"]
BANDS_HPAR: list[str] = [
    "C1",
    "C2",
    "C3",
    "M0",
    "S1",
    "S2",
    "S3",
    "STD",
]  # not possible to add to yaml file since is a ("a", "v") type
BANDS_PLIA: str = "MPLIA"


# pre-processing
def prepare_dc(
    items: ItemCollection,
    bbox: tuple[float, float, float, float],
    bands: str | list[str],
) -> xr.Dataset:
    """Prepare a datacube from the items."""
    return odc_stac.load(
        items,
        bands=bands,
        chunks=CHUNKS,
        bbox=bbox,
        groupby=GROUPBY,
    )


# processing
def process_sig0_dc(
    sig0_dc: xr.Dataset,
    items_sig0: ItemCollection,
    bands: str | list[str],
) -> tuple[xr.Dataset, np.ndarray]:
    """Process the sig0 datacube."""
    sig0_dc: xr.Dataset = (
        post_process_eodc_cube(sig0_dc, items_sig0, bands)
        .rename_vars({"VV": "sig0"})
        .assign_coords(orbit=("time", extract_orbit_names(items_sig0)))
        .dropna(dim="time", how="all")
        .sortby("time")
    )
    orbit_sig0: np.ndarray = order_orbits(sig0_dc)
    sig0_dc: xr.Dataset = sig0_dc.groupby("time").mean(skipna=True)
    sig0_dc: xr.Dataset = sig0_dc.assign_coords(orbit=("time", orbit_sig0))
    sig0_dc: xr.Dataset = sig0_dc.persist()
    return sig0_dc, orbit_sig0


def order_orbits(sig0_dc: xr.Dataset) -> np.ndarray:
    """Order orbits in the sig0 datacube."""
    if sig0_dc.time.shape != ():
        __, indices = np.unique(sig0_dc.time, return_index=True)
        indices.sort()
        return sig0_dc.orbit[indices].data
    return np.array([sig0_dc.orbit.data])


def process_datacube(
    datacube: xr.Dataset,
    items_dc: ItemCollection,
    orbit_sig0: np.ndarray,
    bands: str | list[str],
) -> xr.Dataset:
    """Process the datacube."""
    datacube: xr.Dataset = post_process_eodc_cube(
        datacube,
        items_dc,
        bands,
    ).rename(
        {"time": "orbit"},
    )
    datacube["orbit"] = extract_orbit_names(items_dc)
    datacube = datacube.groupby("orbit").mean(skipna=True)
    datacube = datacube.sel(orbit=orbit_sig0)
    return datacube.persist()


# post-processing
def post_process_eodc_cube(
    dc: xr.Dataset,
    items: ItemCollection,
    bands: str | list[str],
) -> xr.Dataset:
    """Post-process the EODC datacube."""
    if isinstance(bands, str):
        bands: tuple[str] = (bands,)
    for i in bands:
        dc[i] = post_process_eodc_cube_(dc[i], items, i)
    return dc


def post_process_eodc_cube_(
    dc: xr.DataArray,
    items: ItemCollection,
    band: str,
) -> xr.DataArray:
    """Post-process a single band of the EODC datacube."""
    extra_fields = items[0].assets[band].extra_fields.get("raster:bands", [])
    scale = extra_fields[0]["scale"]
    nodata = extra_fields[0]["nodata"]
    # Apply the scaling and nodata masking logic
    return dc.where(dc != nodata) / scale


def extract_orbit_names(items: ItemCollection) -> np.ndarray:
    """Extract orbit names from the items."""
    return np.array(
        [
            items[i].properties["sat:orbit_state"][0].upper()
            + str(items[i].properties["sat:relative_orbit"])
            for i in range(len(items))
        ],
    )


def post_processing(
    dc: xr.Dataset,
    *,
    keep_masks: bool = False,
) -> xr.DataArray:
    """Post-process the datacube to create the flood extent."""
    dc["mask_exceeding_PLIA"] = mask_exceeding_PLIA(dc)
    dc["mask_conflicting_distributions"] = mask_conflicting_distributions(dc)
    dc["mask_outliers"] = mask_outliers(dc)
    dc["mask_denial_high_uncertainty"] = mask_denial_high_uncertainty(dc)
    dc["extent"] = (
        dc.decision
        * dc["mask_exceeding_PLIA"]
        * dc["mask_conflicting_distributions"]
        * dc["mask_outliers"]
        * dc["mask_denial_high_uncertainty"]
    )
    dc["extent"] = remove_speckles(dc.extent)
    if keep_masks:
        dc["extent"] = reduce_masks(dc)
    return dc.extent


def reduce_masks(dc: xr.Dataset) -> tuple[tuple[str, str, str], np.ndarray]:
    """Reduce the masks to a single extent array."""
    return (
        ("time", "latitude", "longitude"),
        np.maximum.reduce(
            [
                dc.extent,
                ~dc.mask_denial_high_uncertainty * 2,
                ~dc.mask_conflicting_distributions * 3,
                ~dc.mask_exceeding_PLIA * 4,
                ~dc.mask_outliers * 5,
            ],
        ),
    )


def mask_denial_high_uncertainty(dc: xr.Dataset) -> xr.DataArray:
    """Mask the datacube where the uncertainty is too high."""
    threshold: float = 0.2
    return np.minimum(dc.nf_post_prob, dc.f_post_prob) < threshold


def mask_conflicting_distributions(dc: xr.Dataset) -> xr.DataArray:
    """Mask the datacube where conflicting distributions are present."""
    return dc.hbsc > (dc.wbsc + 0.5 * 2.754041)


def mask_outliers(dc: xr.Dataset) -> xr.DataArray:
    """Mask the datacube where outliers are present."""
    land_bsc_lower: xr.DataArray = dc.hbsc - 3 * dc.STD
    land_bsc_upper: xr.DataArray = dc.hbsc + 3 * dc.STD
    water_bsc_upper: xr.DataArray = dc.wbsc + 3 * 2.754041
    mask_land_outliers: xr.DataArray = np.logical_and(
        dc.sig0 > land_bsc_lower,
        dc.sig0 < land_bsc_upper,
    )
    mask_water_outliers: xr.DataArray = dc.sig0 < water_bsc_upper
    return mask_land_outliers | mask_water_outliers


def mask_exceeding_PLIA(dc: xr.Dataset) -> xr.DataArray:  # noqa: N802
    """Mask the datacube where the PLIA exceeds the threshold."""
    lower: float = 27
    upper: float = 48
    return np.logical_and(upper >= dc.MPLIA, lower <= dc.MPLIA)


def remove_speckles(
    flood_output: xr.Dataset,
    window_size: int = 5,
) -> xr.Dataset:
    """Apply a rolling median filter.

    Apply a rolling median filter to smooth the
    dataset spatially over longitude
    and latitude.
    """
    flood_output: xr.Dataset = (
        flood_output.rolling({"x": window_size, "y": window_size}, center=True)
        .median(skipna=True)
        .persist()
    )
    return flood_output


def reproject_equi7grid(
    dc: xr.DataArray,
    bbox: tuple[float, float, float, float],
    target_epsg: str = CRS,
) -> xr.DataArray:
    """Reproject the datacube to target EPSG and clip to the bounding box."""
    return ODCExtensionDa(dc).reproject(target_epsg).rio.clip_box(*bbox)
