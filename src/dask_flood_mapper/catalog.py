import pystac_client
from dask_flood_mapper.stac_config import load_config

config = load_config()


def initialize_catalog():
    eodc_catalog = pystac_client.Client.open(config["api"])
    return eodc_catalog


def initialize_search(eodc_catalog, bbox, time_range):
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
