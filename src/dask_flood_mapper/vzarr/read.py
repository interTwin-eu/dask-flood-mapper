import fsspec

import xarray as xr

from dask.distributed import Client
from dask.distributed import WorkerPlugin

class ZstdPlugin(WorkerPlugin):
    def setup(self, worker):
        from imagecodecs import numcodecs # noqa
        from imagecodecs.numcodecs import Zstd # noqa
        numcodecs.register_codec(Zstd)

def install_zstd_plugin_to_client(client: Client):
    client.register_worker_plugin(ZstdPlugin())
    return client

def open_s1_datacube(
        zip_path: str,
        chunks: dict = {
            'X': 15000,
            'Y': 750,
            'polarization': 1,
            "obs": -1,
            "orbit": 1,
            "tile": 1
        }
):
    from imagecodecs import numcodecs # noqa
    from imagecodecs.numcodecs import Zstd # noqa
    numcodecs.register_codec(Zstd)
    mapper = fsspec.get_mapper(
        "reference://",
        fo=f"zip::{zip_path}",
        target_protocol="file",
        remote_protocol="https",
    )

    s1_ds = xr.open_zarr(mapper,
                          consolidated=False,
                          zarr_format=2,
                          chunks={'X': chunks['X'], 'Y': chunks['Y'], 'polarization': chunks['polarization'], "obs": chunks['obs']}
                          ).chunk({'orbit': chunks['orbit'], 'tile': chunks['tile']})
    s1_ds['X'].load()
    s1_ds['Y'].load()
    return s1_ds

