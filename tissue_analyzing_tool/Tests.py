# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 17:26:22 2021

@author: Shahar Kasirer

Small tests to debug image processing code
"""
import matplotlib.pyplot as plt

from basic_image_manipulations import *
from surface_projection import *
import os
from tissue_info import *


test_dir = "test_images"
# test_file = "utricle_small_portion_e17_5.tif"
test_file = "projection.tiff"
test_path = os.path.join(test_dir, test_file)


def adjust_and_save(path):
    movie, axes, shape, metadata = read_tiff(path)
    data_type = movie.dtype
    movie, metadata = set_brightness(movie, axes, metadata, method='bestFit', clearExtreamPrecentage=1)
    
    save_tiff(path.replace('.tif', 'adjusted.tif'), movie, metadata=metadata, axes=axes, data_type=data_type)
    return


def loading_test_image():
     image, axes, shape, metadata = read_tiff(test_path)
     adjusted = set_brightness(image, axes)
     return adjusted


def draw_neighbors(tissue):
     for index, cell in tissue.cells_dfs[0].iterrows():
         centroidx = cell.cx
         centroidy = cell.cy
         for neighbor_label in list(cell.neighbors):
             neighbor = tissue.cells_dfs[0].iloc[neighbor_label-1]
             neighbor_centroidx = neighbor.cx
             neighbor_centroidy = neighbor.cy
             plt.plot((centroidx, neighbor_centroidx), (centroidy, neighbor_centroidy), linewidth = .3, color= 'blue')


def add_numbers(ax, tissue):
    
    for label in tissue.cells.keys():
         cell = tissue.get_cell(label) 
         centroidy, centroidx = cell.get_centroid() 
         ax.text(centroidx, centroidy, str(label), fontsize = 5)
         
    
if __name__ == "__main__":
    input_dir = "D:\\Kasirer\\experimental_results\\fixed\\Vestibule\\zo_myo_dapi\\E15.5\\"
    input_file_name = "21-04-20_E15.5_utricle_zo_myo_dapi_Airyscan Processing_Stitch.tif"
    large_image_projection(input_dir, input_dir, input_file_name, position=0, reference_channel=0, chunk_size=0,
                           bin_size=10, channels_shift=0, min_z=8, max_z=40, method="multi_channel")
