"""Catalog initialization and search functions for Dask Flood Mapper."""

import datetime as dt

from dateutil import parser
from dateutil.relativedelta import relativedelta
from pystac_client import Client, ItemSearch

from dask_flood_mapper.stac_config import load_config

config = load_config()


def initialize_catalog() -> Client:
    """Initialize the EODC catalog client."""
    return Client.open(config["api"])


def initialize_search(
    eodc_catalog: Client,
    bbox: tuple[float, float, float, float],
    time_range: str,
    *,
    dynamic: bool = False,
) -> ItemSearch:
    """Initialize a search for Sentinel-1 data in the EODC catalog."""
    if dynamic:
        time_range = extent_range(eodc_catalog, time_range)
    return eodc_catalog.search(
        collections="SENTINEL1_SIG0_20M",
        bbox=bbox,
        datetime=time_range,
    )


def search_parameters(
    eodc_catalog: Client,
    bbox: tuple[float, float, float, float],
    collections: list[str] | str,
) -> ItemSearch:
    """Search for Sentinel-1 data in the EODC catalog."""
    return eodc_catalog.search(
        collections=collections,  # "SENTINEL1_HPAR" or "SENTINEL1_MPLIA"
        bbox=bbox,
    )


def extent_range(
    eodc_catalog: Client,
    time_range: str,
    years: int = 1,
) -> str:
    """Adjust the time range."""
    search: ItemSearch = eodc_catalog.search()
    split_time_range: list[str] = time_range.split("/")  # type: ignore
    if len(split_time_range) == 1:
        split_time_range: tuple[str, str | None] = search._to_isoformat_range(  # noqa
            time_range,
        )
    delta_time: dt.datetime = parser.parse(
        split_time_range[0],
    ) - relativedelta(years=years, seconds=-1)
    start = search._to_utc_isoformat(delta_time)  # noqa: SLF001
    if split_time_range[1] is not None:
        fmt_datetime: str | None = search._format_datetime(split_time_range)  # noqa
        end: str = fmt_datetime.split("/")[1] if fmt_datetime else ""
    else:
        end: str = split_time_range[0]
    return start + "/" + end


def format_datetime_for_xarray_selection(
    search: ItemSearch,
    time_range: str,
) -> tuple[dt.datetime, ...]:
    """Format the datetime for xarray selection."""
    fmt_datetime: str | None = search._format_datetime(time_range)  # noqa
    if fmt_datetime is None:
        msg: str = "The provided time range is not in the correct format."
        raise ValueError(msg)
    split_time_range: list[str] = fmt_datetime.split("/")
    return tuple(parser.parse(i, ignoretz=True) for i in split_time_range)
