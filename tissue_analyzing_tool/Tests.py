# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 17:26:22 2021

@author: Shahar Kasirer

Small tests to debug image processing code
"""
from basic_image_manipulations import *
import os
from matplotlib import pyplot as plt
from tissue_info import *

test_dir = "D:\\Anastasia\\Fixed_samples"
# test_file = "Local Z Projection of 21_03_04_E17_utricle_Airyscan Processing_Stitch.tif"
test_file= "testimage.tif"
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
     for cell in test_tissue.cells.values():
         centroidy, centroidx = cell.get_centroid()
         for neighbor_label in cell.neighbors:
             neighbor = test_tissue.get_cell(neighbor_label)
             neighbor_centroidy, neighbor_centroidx = neighbor.get_centroid()
             plt.plot((centroidx, neighbor_centroidx), (centroidy, neighbor_centroidy), linewidth = .3, color= 'blue')
def add_numbers(ax, tissue):
    
    for label in tissue.cells.keys():
         cell = tissue.get_cell(label) 
         centroidy, centroidx = cell.get_centroid() 
         ax.text(centroidx, centroidy, str(label), fontsize = 5)
         
     
if __name__ == "__main__":
    image = loading_test_image()
    skeleton, segmented = watershed_segmentation(image[1], .6, 101, 5)
    test_tissue = Tissue(segmented)
    test_tissue.calculate_cellinfo()
    
    fig, ax = plt.subplots(1)
    seg_copy = np.copy(segmented)
    seg_copy[seg_copy > 0] = 1
    ax.imshow(seg_copy, cmap='gray')
    
    test_tissue.get_cell_types(image[0])
    #test_tissue.plot_cell_types()
    #draw_neighbors(test_tissue)
    
    