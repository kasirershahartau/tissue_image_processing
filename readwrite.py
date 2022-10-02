""" functions for reading and writing"""


import numpy as np
from PIL import Image
import dask.array as da
import zarr
from dexp.datasets import ZDataset
from tqdm import tqdm
from aicsimageio import AICSImage
import glob
from typing import Iterator, Tuple
import math
import os
from itertools import product


def convert_czi_to_ZDataset(path_to_czi_dir, path_to_new_zarr, namestr=r'*.czi', prefix=r'timeseries(', suffix=r').czi', chunk_sizes=(1, 64, 256, 256)):
    """function to read in czi files with aicsimageio, extract dask array, then save as zarr with prescribed chunking.
    Uses crude specification of file names, specific to z1 output."""

    filenames = glob.glob(os.path.join(path_to_czi_dir,  namestr))
    ds = ZDataset(path_to_new_zarr, mode="w-")
    for t,filename in tqdm(enumerate(filenames)):
        img = AICSImage(filename)
        if t == 0:

            # create channels using first timepoint info. from this microscope, both channels have same shape
            ds.add_channel(name='Atoh1', shape=(len(filenames), img.dims.Z, img.dims.Y, img.dims.X), dtype=img.dtype, chunks=chunk_sizes)
            ds.add_channel(name='ZO1', shape=(len(filenames), img.dims.Z, img.dims.Y, img.dims.X), dtype=img.dtype, chunks=chunk_sizes)


        # extract the data
        stack = img.data

        # write the two channels to two different groups
        # Ch0 = MCP
        ch = 0
        grp = 'Atoh1'
        this_stack = stack[0, ch]
        ds.write_stack(grp, t, this_stack)

        # Ch1 = H2b
        ch = 1
        grp = 'ZO1'
        this_stack = stack[0, ch]
        ds.write_stack(grp, t, this_stack)

    # done
    ds.close()



if __name__ == "__main__":
    path_to_czi_dir = r"D:\Kasirer\experimental_results\LightSheet\Olga_07.13.21_E15.5\G1"
    path_to_new_zarr = r"D:\Kasirer\experimental_results\LightSheet\Olga_07.13.21_E15.5\G1\zarr_files.zarr"
    convert_czi_to_ZDataset(path_to_czi_dir, path_to_new_zarr, namestr=r'*.czi', prefix=r'timeseries(', suffix=r').czi',
                            chunk_sizes=(1, 64, 256, 256))
