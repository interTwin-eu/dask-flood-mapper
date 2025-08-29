"""Compute the flood decision and probability using Sentinel-1 data."""

from typing import TYPE_CHECKING

import xarray as xr

from dask_flood_mapper.calculation import (
    bayesian_flood_decision,
    bayesian_flood_probability,
    calc_water_likelihood,
    calculate_flood_dc,
    harmonic_expected_backscatter,
)
from dask_flood_mapper.catalog import (
    config,
    format_datetime_for_xarray_selection,
    initialize_catalog,
    initialize_search,
    search_parameters,
)
from dask_flood_mapper.harmonic_params import (
    create_harmonic_parameters,
    process_harmonic_parameters_datacube,
)
from dask_flood_mapper.processing import (
    BANDS_HPAR,
    post_processing,
    prepare_dc,
    process_datacube,
    process_sig0_dc,
    reproject_equi7grid,
)

if TYPE_CHECKING:
    import datetime as dt

    from pystac.item_collection import ItemCollection
    from pystac_client import Client, ItemSearch

# import parameters from config.yaml file
crs = config["base"]["crs"]
chunks = config["base"]["chunks"]
groupby = config["base"]["groupby"]
BANDS_SIG0: str = "VV"
BANDS_PLIA: str = "MPLIA"


def decision(
    bbox: tuple[float, float, float, float],
    datetime: str,
    *,
    dynamic: bool = False,
    keep_masks: bool = False,
) -> xr.DataArray:
    """Bayesian Flood Decision.

    Classify Sentinel-1 radar images by simple Bayes inference into flood (1)
    and non-flood (0). Besides radar images, this algorithm relies on two other
    datasets stored at the Earth Observation Data Centre For Water Resources
    Monitoring (EODC); harmonic parameters based on a fit on per land pixel
    timeseries and the projected incidence angle of the measurement. The latter
    two datasets are required to calculate the land and water likelihood
    distributions, respectively.

    Parameters
    ----------
    bbox : tuple of float or tuple of int
        Geographic bounding box, consisting of minimum longitude, minimum
        latitude, maximum longitude, maximum latitude
    datetime: string
        Datetime string:

          - A closed range: "2022-10-01/2022-10-07"
          - Whole month, year or day: "2022-01"
          - Open range with current date: "2022-01-01/.."
          - Specific time instance: "2022-01-01T05:34:46"
    dynamic: bool, default=False
        Calculate harmonic parameters dynamically for the selected bbox by
        fitting a harmonic function per pixel to a three year span of sigma
        nought before the lower limit of the datetime provided. This operation
        is computationally demanding.

    keep_masks: bool, default=False
        If True, the masks for water and land likelihood are kept in the output
        datacube. If False, only the flood decision and probabilities are kept.

    Returns
    -------
        flood decision : xarray.DataArray of 0 (non-flood) and 1 (flood)

    See Also
    --------
    probability

    Examples
    --------
    >>> from dask_flood_mapper import flood
    >>>
    >>>
    >>> time_range = "2022-10-11/2022-10-25"
    >>> bbox = [12.3, 54.3, 13.1, 54.6]
    >>> flood.decision(bbox=bbox, datetime=time_range).compute()
    sigma naught datacube processed
    harmonic parameter datacube processed
    projected local incidence angle processed
    <xarray.DataArray 'decision' (time: 8, y: 1048, x: 2793)> Size: 187MB
    array([[[nan, nan, nan, ...,  0., nan, nan],
            [ 0.,  0.,  0., ...,  0., nan, nan],
            [ 0.,  0.,  0., ...,  0., nan, nan],
            ...,
            [nan, nan,  0., ...,  0.,  0.,  0.],
            [nan, nan,  0., ...,  0., nan, nan],
            [nan, nan,  0., ..., nan, nan, nan]],
        ...
           [[nan, nan, nan, ..., nan, nan, nan],
            [nan, nan, nan, ..., nan, nan, nan],
            [nan, nan, nan, ..., nan, nan, nan],
            ...,
            [nan, nan,  0., ...,  0.,  0.,  0.],
            [nan, nan,  0., ...,  0., nan, nan],
            [nan, nan,  0., ..., nan, nan, nan]],
        ...
           [[nan, nan, nan, ..., nan, nan, nan],
            [ 0.,  0.,  0., ..., nan, nan, nan],
            [ 0.,  0.,  0., ..., nan, nan, nan],
            ...,
            [nan, nan,  0., ...,  0.,  0.,  0.],
            [nan, nan,  0., ...,  0., nan, nan],
            [nan, nan,  0., ..., nan, nan, nan]],
        ...
           [[nan, nan, nan, ..., nan, nan, nan],
            [ 0.,  0.,  0., ..., nan, nan, nan],
            [ 0.,  0.,  0., ..., nan, nan, nan],
            ...,
            [nan, nan, nan, ..., nan, nan, nan],
            [nan, nan, nan, ..., nan, nan, nan],
            [nan, nan, nan, ..., nan, nan, nan]],
        ...
           [[nan, nan, nan, ...,  0., nan, nan],
            [ 0.,  0.,  0., ...,  0., nan, nan],
            [ 0.,  0.,  0., ...,  0., nan, nan],
            ...,
            [nan, nan,  0., ...,  0.,  0.,  0.],
            [nan, nan,  0., ...,  0., nan, nan],
            [nan, nan,  0., ..., nan, nan, nan]]])
    Coordinates:
    * x            (x) float64 22kB 12.3 12.3 12.3 12.3 ... 13.1 13.1 13.1 13.1
    * y            (y) float64 8kB 54.6 54.6 54.6 54.6 ... 54.3 54.3 54.3 54.3
    * time         (time) datetime64[ns] 64B 2022-10-11T05:25:01 ... 2022-10-23...
        spatial_ref  int64 8B 0
    Attributes:
        _FillValue:  nan
    >>>

    """  # noqa: E501
    sig0_dc, hpar_dc, plia_dc = preprocess(bbox, datetime, dynamic=dynamic)
    flood_dc: xr.Dataset = calculate_flood_dc(sig0_dc, plia_dc, hpar_dc)
    flood_dc["wbsc"] = calc_water_likelihood(flood_dc)  # Water
    flood_dc["hbsc"] = harmonic_expected_backscatter(flood_dc)  # Land
    flood_dc["decision"] = bayesian_flood_decision(flood_dc)
    flood_dc["f_post_prob"] = bayesian_flood_probability(flood_dc)
    flood_dc["nf_post_prob"] = 1 - flood_dc["f_post_prob"]
    flood_output = post_processing(flood_dc, keep_masks=keep_masks)
    return reproject_equi7grid(flood_output, bbox=bbox)


def probability(
    bbox: tuple[float, float, float, float],
    datetime: str,
    *,
    dynamic: bool = False,
) -> xr.DataArray:
    """Bayesian Flood Probability.

    Classify Sentinel-1 radar images by simple Bayes inference into a
    probability of flood,ranging from 0 (minimum probability of flood) to 1
    (maximum probability of flood).Besides radar images, this algorithm relies
    on two other datasets stored at the Earth Observation Data Centre For Water
    Resources Monitoring (EODC); harmonic parameters based on a fit on per land
    pixel timeseries and the projected incidence angle of the measurement. The
    latter two datasets are required to calculate the land and water likelihood
    distributions, respectively.

    Parameters
    ----------
    bbox : tuple of float or tuple of int
        Geographic bounding box, consisting of minimum longitude, minimum
        latitude, maximum longitude, maximum latitude
    datetime: string
        Datetime string:

          - A closed range: "2022-10-01/2022-10-07"
          - Whole month, year or day: "2022-01"
          - Open range with current date: "2022-01-01/.."
          - Specific time instance: "2022-01-01T05:34:46"
    dynamic: bool, default=False
        Calculate harmonic parameters dynamically for the selected bbox by
        fitting a harmonic function per pixel to a three year span of sigma
        nought before the lower limit of the datetime provided. This operation
        is computationally demanding.

    Returns
    -------
        flood probability : xarray.DataArray ranging from 0 (0% estimation of
        flood) to 1 (100% estimation of flood)

    See Also
    --------
    decision

    Examples
    --------
    >>> from dask_flood_mapper import flood
    >>>
    >>>
    >>> time_range = "2022-10-11/2022-10-25"
    >>> bbox = [12.3, 54.3, 13.1, 54.6]
    >>> flood.probability(bbox=bbox, datetime=time_range).compute()
    sigma naught datacube processed
    harmonic parameter datacube processed
    projected local incidence angle processed
    <xarray.DataArray 'probability' (time: 8, y: 1048, x: 2793)> Size: 187MB
    array([[[1.86211960e-01, 2.15371963e-01, 2.05863488e-01, ...,
         2.52572128e-01, 2.57730876e-01, 2.44652898e-01],
        [1.28253888e-01, 1.51311120e-01, 1.72076672e-01, ...,
         3.41533329e-01, 3.02598322e-01, 2.72460141e-01],
        [1.07656028e-01, 1.07656028e-01, 1.43597848e-01, ...,
         2.92728749e-01, 2.91336553e-01, 2.25547046e-01],
        ...,
        [3.30095422e-01, 3.00706753e-01, 3.38240209e-01, ...,
         4.56804879e-03, 3.38973420e-03, 1.06926495e-02],
        [3.32022998e-01, 3.39287332e-01, 3.61899104e-01, ...,
         2.65790544e-04, 5.72882888e-04, 4.91604904e-04],
        [3.26255229e-01, 3.24131583e-01, 3.31034180e-01, ...,
         1.42086512e-03, 5.99092039e-04, 3.18391829e-04]],
    ...
       [[           nan,            nan,            nan, ...,
                    nan,            nan,            nan],
        [           nan,            nan,            nan, ...,
                    nan,            nan,            nan],
        [           nan,            nan,            nan, ...,
                    nan,            nan,            nan],
    ...
        [           nan,            nan,            nan, ...,
                    nan,            nan,            nan],
        [           nan,            nan,            nan, ...,
                    nan,            nan,            nan],
        [           nan,            nan,            nan, ...,
                    nan,            nan,            nan]],
    ....
       [[3.73245925e-01, 3.48150842e-01, 3.51988605e-01, ...,
         4.23590506e-01, 4.06161145e-01, 4.00064362e-01],
        [3.75232243e-01, 3.30142918e-01, 3.09075590e-01, ...,
         3.78596082e-01, 3.95209483e-01, 3.92170382e-01],
        [3.30411323e-01, 3.30411323e-01, 4.13602532e-01, ...,
         3.89603260e-01, 4.04806821e-01, 4.92500567e-01],
        ...,
        [4.82000282e-01, 4.95535115e-01, 4.80515991e-01, ...,
         1.02485531e-02, 3.87479773e-03, 4.40659361e-03],
        [5.10606062e-01, 5.15671724e-01, 4.71831786e-01, ...,
         3.71247499e-04, 2.30112900e-04, 2.16098949e-04],
        [5.17821688e-01, 5.59295596e-01, 5.43262041e-01, ...,
         3.05140177e-03, 5.17321966e-04, 2.79084525e-04]]])
    Coordinates:
    * x            (x) float64 22kB 12.3 12.3 12.3 12.3 ... 13.1 13.1 13.1 13.1
    * y            (y) float64 8kB 54.6 54.6 54.6 54.6 ... 54.3 54.3 54.3 54.3
    * time         (time) datetime64[ns] 64B 2022-10-11T05:25:01 ... 2022-10-23...
        spatial_ref  int64 8B 0
    Attributes:
        _FillValue:  nan
    >>>

    """  # noqa: E501
    sig0_dc, hpar_dc, plia_dc = preprocess(bbox, datetime, dynamic=dynamic)
    flood_dc = calculate_flood_dc(sig0_dc, plia_dc, hpar_dc)
    flood_dc["wbsc"] = calc_water_likelihood(flood_dc)  # Water
    flood_dc["hbsc"] = harmonic_expected_backscatter(flood_dc)  # Land
    return reproject_equi7grid(bayesian_flood_probability(flood_dc), bbox=bbox)


def preprocess(
    bbox: tuple[float, float, float, float],
    datetime: str,
    *,
    dynamic: bool,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Preprocess the data for flood decision and probability calculation."""
    eodc_catalog: Client = initialize_catalog()
    search: ItemSearch = initialize_search(
        eodc_catalog,
        bbox,
        datetime,
        dynamic=dynamic,
    )

    items_sig0: ItemCollection = search.item_collection()
    sig0_dc: xr.Dataset = prepare_dc(items_sig0, bbox, bands="VV")
    sig0_dc, orbit_sig0 = process_sig0_dc(sig0_dc, items_sig0, bands="VV")
    print("sigma naught datacube processed")  # noqa: T201

    if dynamic:
        datetime_xr: tuple[dt.datetime, ...] = (
            format_datetime_for_xarray_selection(
                search,
                datetime,
            )
        )
        hpar_list: list[tuple[int, xr.DataArray]] = create_harmonic_parameters(
            sig0_dc,
        )
        sig0_dc, hpar_dc, orbit_sig0 = process_harmonic_parameters_datacube(
            sig0_dc,
            datetime_xr,
            hpar_list,
        )
    else:
        search_hpar = search_parameters(
            eodc_catalog,
            bbox,
            collections="SENTINEL1_HPAR",
        )
        items_hpar: ItemCollection = search_hpar.item_collection()
        hpar_dc: xr.Dataset = prepare_dc(items_hpar, bbox, bands=BANDS_HPAR)
        hpar_dc: xr.Dataset = process_datacube(
            hpar_dc,
            items_hpar,
            orbit_sig0,
            BANDS_HPAR,
        )
    print("harmonic parameter datacube processed")  # noqa: T201

    search_plia: ItemSearch = search_parameters(
        eodc_catalog,
        bbox,
        collections="SENTINEL1_MPLIA",
    )
    items_plia: ItemCollection = search_plia.item_collection()
    plia_dc: xr.Dataset = prepare_dc(items_plia, bbox, bands=BANDS_PLIA)
    plia_dc: xr.Dataset = process_datacube(
        plia_dc,
        items_plia,
        orbit_sig0,
        bands="MPLIA",
    )
    print("projected local incidence angle processed")  # noqa: T201

    return sig0_dc, hpar_dc, plia_dc
