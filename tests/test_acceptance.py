from datetime import datetime

import numpy as np
import rioxarray  # noqa
import xarray as xr
from dask_flood_mapper import flood


def assert_datacube_eq(actual, expected):
    xr.testing.assert_allclose(actual, expected, rtol=0.01)


def make_datacube(values, y, x, time):
    values_arr = np.array(values).astype(np.float32)
    return xr.DataArray(
        values_arr,
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": y, "longitude": x, "spatial_ref": 0},
    )


class TestFloodMapNorthernGermany2022:
    time_range = "2022-10-11T05:25:26"
    minlon, maxlon = 12.999, 13
    minlat, maxlat = 53.999, 54
    bounding_box = [minlon, minlat, maxlon, maxlat]
    flood_map = flood.decision(bbox=bounding_box, datetime=time_range)

    def test_that_flood_map_is_xarray(self):
        assert isinstance(self.flood_map, xr.DataArray)

    def test_that_flood_map_is_created(self):
        assert_datacube_eq(
            self.flood_map,
            make_datacube(
                [
                    [
                        [np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, 0.0, 0.0, np.nan],
                        [np.nan, np.nan, np.nan, 0.0, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan],
                    ]
                ],
                x=[13.0, 13.0, 13.0, 13.0, 13.0],
                y=[54.0, 54.0, 54.0, 54.0, 54.0, 54.0],
                time=[datetime.strptime(self.time_range, "%Y-%m-%dT%H:%M:%S")],
            ),
        )


# class TestFloodMapIsOutOfDateRange:
#     minlon, maxlon = 12.3, 13.1
#     minlat, maxlat = 54.3, 54.6
#     bounding_box = [minlon, minlat, maxlon, maxlat]

#     def test_that_datetime_is_too_specific(self):
#         with pytest.raises(ValueError):
#             flood.decision(
# bbox=self.bounding_box, datetime="2022-10-11:20:22"
# )

#     def test_that_datetime_is_out_of_range(self):
#         with pytest.raises(ValueError):
#             flood.decision(
# bbox=self.bounding_box, datetime="2011-10-11/2011-10-25"
# )
