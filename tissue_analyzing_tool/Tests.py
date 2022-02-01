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
    # shape = get_image_dimensions(test_path)
    # projection = np.zeros((shape.T, shape.C, 1, shape.Y, shape.X))
    # for i,chunk in enumerate(read_image_in_chunks(test_path, 1000, 1000, 0, 0, 0, surface_projection, projection,
    #                                             "TCZXY", 0, 0, 102, 'max_std', 5)):
    #     print("Chunk %d had been projected" % i)
    # save_tiff(os.path.join(test_dir, 'projection_by_chunks.tiff'), projection,
    #           data_type="uint16")

    # image, axes, shape, metadata = read_tiff(test_path)
    # adjusted = set_brightness(image, axes)
    # projection = surface_projection(image, axes, 0, 0, 102, 'max_std', 10)
    # save_tiff(os.path.join(test_dir, 'projection.tiff'), projection.swapaxes(-1,-2), metadata=metadata, data_type="uint16")

    # image = loading_test_image()
    # small_zo_image = image[0]
    # segmentation = watershed_segmentation(small_zo_image, 0.3*np.max(small_zo_image), 3)
    # plt.imshow(segmentation)
    # tissue = Tissue()
    # tissue.setLabels(1, segmentation)
    # tissue.calculate_frame_cellinfo(1)
    # draw_neighbors(tissue)
    # plt.show()
    directory = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2021-12-26_P0_utricle_ablation\\"
    file_names = ["initial_Out.czi", "after1_Out.czi", "after2_Out.czi", "after3_Out.czi", "after4_Out.czi",
                  "after5_Out.czi", "after6_Out.czi", "after7_Out.czi"]
    paths = [os.path.join(directory, file) for file in file_names]
    movie_surface_projection([paths[0]], (8,6,6,8), 4)
