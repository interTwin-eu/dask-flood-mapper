import pystac_client
from dask_flood_mapper.stac_config import load_config
from dateutil import parser
from dateutil.relativedelta import relativedelta

config = load_config()


def initialize_catalog():
    eodc_catalog = pystac_client.Client.open(config["api"])
    return eodc_catalog


def initialize_search(eodc_catalog, bbox, time_range, dynamic):
    if dynamic:
        time_range = extent_range(eodc_catalog, time_range)
    search = eodc_catalog.search(
        collections="SENTINEL1_SIG0_20M",
        bbox=bbox,
        datetime=time_range,
    )
    return search


def search_parameters(eodc_catalog, bbox, collections):
    search = eodc_catalog.search(
        collections=collections,  # "SENTINEL1_HPAR" or "SENTINEL1_MPLIA"
        bbox=bbox,
    )

    return search


def extent_range(eodc_catalog, time_range, years=1):
    search = eodc_catalog.search()
    split_time_range = time_range.split("/")
    if len(split_time_range) == 1:
        split_time_range = search._to_isoformat_range(time_range)
    delta_time = parser.parse(split_time_range[0]) - relativedelta(
        years=years, seconds=-1
    )
    start = search._to_utc_isoformat(delta_time)
    if split_time_range[1] is not None:
        end = search._format_datetime(split_time_range).split("/")[1]
    else:
        end = split_time_range[0]
    return start + "/" + end


def format_datetime_for_xarray_selection(search, time_range):
    split_time_range = search._format_datetime(time_range).split("/")
    return [parser.parse(i, ignoretz=True) for i in split_time_range]
