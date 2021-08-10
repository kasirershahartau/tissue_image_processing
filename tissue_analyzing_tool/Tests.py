# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 17:26:22 2021

@author: Shahar Kasirer

Small tests to debug image processing code
"""
from basic_image_manipulations import *
from surface_projection import *
import os
from matplotlib import pyplot as plt
from tissue_info import *
test_dir = "C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\experimental_results\\fixed\\Vestibule\\zo_myo_dapi\\E17.5"
test_file = "21-04-13_E17.5_utricle2_zo1_atho1_dapi_Airyscan Processing.czi"
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
    dims = get_image_dimensions(test_path)
    projection = np.zeros((3, dims.X, dims.Y))
    x = 0
    y = 0
    dx = 1000
    dy = 1000
    for chunk in read_image_in_chunks(test_path, dx = dx, dy = dy, dz = 0, dc = 0, dt= 1):
        chunk = set_brightness(chunk, "TCZXY", clearExtreamPrecentage=1)
        chunk_x = min(x+dx, dims.X) - x
        chunk_y = min(y+dy, dims.Y) - y
        chunk_projection = surface_projection(chunk.reshape((3, 20, chunk_x, chunk_y)), "CZXY", 1, 0, 4, "max_std", 5)
        projection[:,x:min(x+dx, dims.X), y:min(y+dy, dims.Y)] = chunk_projection
        if x+dx > dims.X:
            x = 0
        if y+dy > dims.Y:
            y = 0
            x += dx
        y += dy
        
    save_tiff(test_path.replace(".czi","_projected.tif"), projection, axes="CXY", data_type="uint16") 


