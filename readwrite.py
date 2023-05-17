""" functions for reading and writing"""


import numpy as np
from PIL import Image
import dask.array as da
import zarr
from dexp.datasets import ZDataset
from tqdm import tqdm
from aicsimageio import AICSImage
from aicsimageio.readers import czi_reader, bioformats_reader
import glob
from typing import Iterator, Tuple
import math
import os
from itertools import product


def convert_czi_to_ZDataset(path_to_czi_dir, path_to_new_zarr, namestr=r'*.czi', prefix=r'timeseries(', suffix=r').czi', chunk_sizes=(1, 64, 256, 256), series_num=1):
    """function to read in czi files with aicsimageio, extract dask array, then save as zarr with prescribed chunking.
    Uses crude specification of file names, specific to z1 output."""
    base_path_to_new_zarr = path_to_new_zarr
    for series in range(series_num):
        if series_num>1:
            path_to_new_zarr = base_path_to_new_zarr.replace(".zarr","series_%d.zarr"%series)
        filenames = glob.glob(os.path.join(path_to_czi_dir,  namestr))
        ds = ZDataset(path_to_new_zarr, mode="w-")

        # First pass - getting image size data
        X_max = 0
        Y_max = 0
        Z_max = 0

        T_total = 0
        for i, filename in tqdm(enumerate(filenames)):
            img = AICSImage(filename, reader=bioformats_reader.BioformatsReader)
            img.set_scene(series)
            T_total += img.dims.T
            X_max = max(X_max, img.dims.X)
            Y_max = max(Y_max, img.dims.Y)
            Z_max = max(Z_max, img.dims.Z)
        t = 0
        for i ,filename in tqdm(enumerate(filenames)):
            img = AICSImage(filename, reader=bioformats_reader.BioformatsReader)
            img.set_scene(series)
            if i == 0:
                # create channels using first timepoint info. from this microscope, both channels have same shape
                ds.add_channel(name='Atoh1', shape=(T_total, Z_max, Y_max, X_max), dtype=img.dtype, chunks=chunk_sizes)
                # ds.add_channel(name='ZO1', shape=(len(filenames), img.dims.Z, img.dims.Y, img.dims.X), dtype=img.dtype, chunks=chunk_sizes)

            # extract the data
            stack = img.get_image_dask_data()
            for j in range(img.dims.T):
                # write the two channels to two different groups
                # Ch0 = MCP
                ch = 0
                grp = 'Atoh1'
                this_stack = np.zeros((Z_max, Y_max, X_max))
                this_stack[:img.dims.Z, :img.dims.Y, :img.dims.X] = stack[j, ch].compute()
                ds.write_stack(grp, t, this_stack)
                t += 1
                # Ch1 = H2b
                # ch = 1
                # grp = 'ZO1'
                # this_stack = stack[0, ch]
                # ds.write_stack(grp, t, this_stack)

        # done
        ds.close()



if __name__ == "__main__":
    path_to_czi_dir = r"D:\Kasirer\experimental_results\LightSheet\2023-02-07 E15.5 Atoh only"
    path_to_new_zarr = r"D:\Kasirer\experimental_results\LightSheet\2023-02-07 E15.5 Atoh only\zarr_files.zarr"
    convert_czi_to_ZDataset(path_to_czi_dir, path_to_new_zarr, namestr=r'*.czi', prefix=r'timeseries(', suffix=r').czi',
                            chunk_sizes=(1, 64, 256, 256), series_num=2)
