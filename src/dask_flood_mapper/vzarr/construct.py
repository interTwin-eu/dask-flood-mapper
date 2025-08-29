import base64
import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import fsspec
import numpy as np
import pandas as pd
import tifffile
import ujson
import xarray as xr
import zarr
from fsspec.implementations.reference import LazyReferenceMapper
from tqdm import tqdm


def generate_equi7_vzarr_from_dataframe(
    paths_df: pd.DataFrame,
    outfile: Path,
    replace_in_paths: Tuple[str, str] | None = None,
    metadata: dict = {},
):
    paths_df = paths_df.sort_index()

    time_coords, paths_df["time_idx"] = np.unique(paths_df.index, return_inverse=True)
    paths_df[["lat_coord", "lon_coord"]] = paths_df.apply(
        (lambda x: (int(x["tile_name"][1:4]), int(x["tile_name"][5:8]))),
        axis=1,
        result_type="expand",
    )
    _, paths_df["y_idx"] = np.unique(paths_df["lat_coord"], return_inverse=True)
    _, paths_df["x_idx"] = np.unique(paths_df["lon_coord"], return_inverse=True)
    polarization_coords, paths_df["polarization_idx"] = np.unique(
        paths_df["band"], return_inverse=True
    )
    orbit_coords, paths_df["orbit_idx"] = np.unique(
        paths_df["extra_field"], return_inverse=True
    )
    tile_coords, paths_df["tile_idx"] = np.unique(
        paths_df["tile_name"], return_inverse=True
    )
    X_coords, Y_coords = coordinates_from_e7_string(tile_coords, n_pixels=15000)

    # Pre-assign sequential order to all files
    file_order_map = {}
    file_count = 0

    # It's useful to group by tile first, assuming it's more likely that we'll be
    # reading many orbits from a single tile than many tiles from a single orbit.
    grouper = paths_df.groupby(
        [
            "tile_idx",
            "orbit_idx",
            "polarization_idx",
        ]
    )

    for i, (group, group_df) in enumerate(grouper):
        files = group_df.sort_index()["filepath"].to_list()
        files = [str(path) for path in files]
        for j, filename in enumerate(files):
            file_order_map[filename] = file_count
            file_count += 1

    fs, _ = fsspec.core.url_to_fs(outfile, **({}))
    out_refs = LazyReferenceMapper.create(
        record_size=300000,
        root=outfile,
        fs=fs,
        categorical_threshold=10,
    )

    coordinates = {}
    zarrgroup = zarr.open_group(coordinates)

    zarrgroup.array(
        "tile",
        data=tile_coords,
        dtype="<U11",  # e.g. E000N000T12
        fill_value="",
    ).attrs["_ARRAY_DIMENSIONS"] = ["tile"]

    zarrgroup.array(
        "polarization",
        data=polarization_coords,
        dtype="<U2",  # e.g. VV, VH
        fill_value="",
    ).attrs["_ARRAY_DIMENSIONS"] = ["polarization"]

    zarrgroup.array(
        "orbit",
        data=orbit_coords,
        dtype="<U4",  # e.g. A000, D123
        fill_value="",
    ).attrs["_ARRAY_DIMENSIONS"] = ["orbit"]

    # each tile has its own X and Y coordinates
    zarrgroup.array(
        "X",
        data=X_coords,
        dtype="int32",
        fill_value=-9999,
    ).attrs[
        "_ARRAY_DIMENSIONS"
    ] = ["X", "tile"]

    zarrgroup.array(
        "Y",
        data=Y_coords,
        dtype="int32",
        fill_value=-9999,
    ).attrs[
        "_ARRAY_DIMENSIONS"
    ] = ["Y", "tile"]

    time_coords = grouper_to_da(grouper)

    # each tile/orbit combination has its own time coordinates
    # along the observation dimension.
    zarrgroup.array(
        "time",
        data=time_coords.values,
        dtype="<M8[s]",
    ).attrs[
        "_ARRAY_DIMENSIONS"
    ] = ["obs", "tile", "orbit"]

    zarr_shape = [
        len(time_coords.obs),
        int(paths_df["tile_idx"].max()) + 1,
        int(paths_df["orbit_idx"].max()) + 1,
        int(paths_df["polarization_idx"].max()) + 1,
    ]

    all_tasks = []
    for i, (group, group_df) in enumerate(grouper):
        files = group_df.sort_index()["filepath"].to_list()
        for j, filename in enumerate(files):
            zarr_base_idx = [int(idx) for idx in group]
            zarr_idx = [j, *zarr_base_idx]
            all_tasks.append((filename, zarr_idx))

    def process_single_file(task, append_mode=True):
        filename, zarr_idx = task
        url, name = str(filename).rsplit("/", 1)

        try:
            with fs.open(filename) as fh:
                with tifffile.TiffFile(fh, name=name) as tif:
                    with tif.series[0].aszarr(
                        chunkmode=0, zattrs={"scale_factor": 0.1, **metadata}
                    ) as store:
                        temp_buffer = io.StringIO()
                        store.write_fsspec(
                            temp_buffer,
                            url=url,
                            _shape=zarr_shape,
                            _axes=["obs", "tile", "orbit", "polarization"],
                            _index=zarr_idx,
                            _append=append_mode,
                            groupname="sig0",
                            _close=False,
                        )
                        return temp_buffer.getvalue()
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            return None

    batch_size = 100
    with tqdm(total=len(all_tasks), desc="Processing TIFF files", unit="file") as pbar:
        # Process the FIRST file separately with append=False
        if all_tasks:
            first_task = all_tasks[0]
            remaining_tasks = all_tasks[1:]

            pbar.set_description("Processing first file (header)")
            first_result = process_single_file(first_task, append_mode=False)
            if first_result:
                refs = ujson.loads(first_result + "}")
                for k in sorted(refs):
                    out_refs[k] = refs[k]
            pbar.update(1)

            # Now process remaining files in parallel batches with append=True
            batches = [
                remaining_tasks[i : i + batch_size]
                for i in range(0, len(remaining_tasks), batch_size)
            ]

            for batch_idx, batch in enumerate(batches):
                pbar.set_description(f"Processing batch {batch_idx + 1}/{len(batches)}")

                with ThreadPoolExecutor(max_workers=14) as executor:
                    # All remaining files use append=True
                    future_to_task = {
                        executor.submit(
                            process_single_file, task, append_mode=True
                        ): task
                        for task in batch
                    }

                    # Collect results as they complete
                    batch_results = []
                    for future in as_completed(future_to_task):
                        try:
                            result = future.result()
                            if result:
                                batch_results.append(result)
                        except Exception as e:
                            logging.error(f"Failed to process file: {e}")
                        finally:
                            pbar.update(1)

                # Write batch results
                for result in batch_results:
                    if result:
                        refs = ujson.loads("{" + result[1:] + "}")
                        for k in sorted(refs):
                            out_refs[k] = refs[k]
                        out_refs.flush()

    # base64 encode any values containing non-ascii characters
    for k, v in coordinates.items():
        try:
            coordinates[k] = v.decode()
        except UnicodeDecodeError:
            coordinates[k] = "base64:" + base64.b64encode(v).decode()

    coordinates_refs = ujson.loads(
        tifffile.ZarrStore._json(coordinates).decode()
    )  # ignore preceding stuff
    for k in sorted(coordinates_refs):
        out_refs[k] = coordinates_refs[k]
    out_refs.flush()

    refs = Path(outfile)

    if replace_in_paths is not None:
        for pf in refs.glob("sig0/*.parq"):
            chunk_df = pd.read_parquet(pf)
            chunk_df["path"] = chunk_df.path.apply(
                lambda p: np.nan
                if p is np.nan
                else p.replace(replace_in_paths[0], replace_in_paths[1])
            )
            chunk_df.to_parquet(pf, engine="fastparquet")

    mapper = fsspec.get_mapper(
        "reference://",
        fo=str(refs),
        target_protocol="file",
        remote_protocol="https",
        asynchronous=False,
    )
    return mapper


def grouper_to_da(grouper):
    # Get the unique keys and convert them to a structured format for easier access
    keys = list(grouper.groups.keys())
    time_dict = {k: [pd.Timestamp(t) for t in grouper.groups[k]] for k in keys}

    # Extract the unique orbit, tile, and obs indices
    orbit_indices = sorted(set(k[1] for k in keys))
    tile_indices = sorted(set(k[0] for k in keys))
    max_obs = max(len(times) for times in time_dict.values())

    # Create a 3D array filled with NaN values
    time_array = np.full(
        (max_obs, len(tile_indices), len(orbit_indices)),
        np.datetime64("NaT"),
        dtype="datetime64[ns]",
    )

    # Fill the array with datetime values
    for key, times in time_dict.items():
        orbit_idx = orbit_indices.index(key[1])
        tile_idx = tile_indices.index(key[0])
        for obs_idx, t in enumerate(times):
            time_array[obs_idx, tile_idx, orbit_idx] = np.datetime64(t)

    # Convert to xarray DataArray for better labeling
    time_da = xr.DataArray(
        time_array,
        dims=["obs", "tile", "orbit"],
        coords={
            "obs": np.arange(max_obs),
            "tile": tile_indices,
            "orbit": orbit_indices,
        },
    )
    return time_da


def coordinates_from_e7_string(e7_strings, n_pixels):
    """
    Convert a list of strings of the form E0N0T0 to arrays of coordinates.
    Returns x_coords, y_coords of shape (n_pixels, len(e7_strings))
    """
    x_coords = np.zeros((n_pixels, len(e7_strings)))
    y_coords = np.zeros((n_pixels, len(e7_strings)))
    for idx, e7_string in enumerate(e7_strings):
        s = e7_string[1:]
        x_start = int(s[:3]) * 100000
        y_start = int(s[4:7]) * 100000
        tile_side_length = int(s[8:]) * 100000
        x_coords[:, idx] = np.linspace(
            x_start,
            x_start + tile_side_length - (tile_side_length / n_pixels),
            n_pixels,
        )
        y_coords[:, idx] = np.linspace(
            y_start,
            y_start + tile_side_length - (tile_side_length / n_pixels),
            n_pixels,
        )
    return x_coords, np.flip(y_coords, axis=0)
