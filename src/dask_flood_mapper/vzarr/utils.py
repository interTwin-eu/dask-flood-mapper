import numpy as np
import xarray as xr

def get_bbox_from_tile_cube(
        ds: xr.Dataset,
        bounding_box: tuple[float, float, float, float],
        y_chunk_size: int = 300
):
    minx, miny, maxx, maxy = bounding_box
    X = ds.X
    Y = ds.Y
    useful_X_tiles = ((X >= minx) & (X <= maxx)).any('X')
    useful_Y_tiles = ((Y >= miny) & (Y <= maxy)).any('Y')
    needed_tiles = useful_X_tiles & useful_Y_tiles
    X = ds.X.isel(tile=needed_tiles)
    Y = ds.Y.isel(tile=needed_tiles)
    needed_X = ((X >= minx) & (X <= maxx)).any('tile')
    needed_Y = ((Y >= miny) & (Y <= maxy)).any('tile')
    needed_Y.data = extend_true_to_chunk_edges(needed_Y.data, y_chunk_size, axis=0)

    tiles_ds = ds.isel(tile=needed_tiles.load())
    out_ds = tiles_ds.isel(Y=needed_Y).isel(X=needed_X)
    valid_images = out_ds.time.notnull()
    out_ds = (out_ds.isel(orbit=valid_images.any(['obs', 'tile']).values,
                         obs=valid_images.any(['orbit', 'tile']).values))
    return out_ds

def extend_true_to_chunk_edges(
        arr: np.ndarray[bool],
        chunk_size: int,
        axis: int = -1
):
    """Extend True values in a boolean array to the edges of chunks of given size along some axis."""

    arr = np.asarray(arr)
    out = np.zeros_like(arr, dtype=bool)

    arr_moved = np.moveaxis(arr, axis, -1)
    out_moved = np.moveaxis(out, axis, -1)

    shape = arr_moved.shape
    n = shape[-1]
    n_chunks = (n + chunk_size - 1) // chunk_size

    it = np.nditer(arr_moved[..., 0], flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        line = arr_moved[idx]  # 1D boolean array

        line_out = np.zeros_like(line, dtype=bool)
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n)
            if np.any(line[start:end]):
                line_out[start:end] = True
        out_moved[idx] = line_out

    return np.moveaxis(out_moved, -1, axis)
