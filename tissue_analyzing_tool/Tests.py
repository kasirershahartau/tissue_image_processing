# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 17:26:22 2021

@author: Shahar Kasirer

Small tests to debug image processing code
"""
from basic_image_manipulations import *
from surface_projection import *
from surface_proj_m import *
from surf import *
from skimage import img_as_float32
import os
import numpy as np
from matplotlib import pyplot as plt
from tissue_info import *
from PIL import Image
import imageio
import cv2
from tifffile import imsave

test_dir = "D:\\DavidS10\\Desktop"
test_file = "21_01_05_utricle_p0_Airyscan Processing_Stitch.czi"
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
    projection = np.zeros((dims.Z, dims.X, dims.Y))
    record = []
    x_start = 0
    y_start = 0
    dx = 1000
    dy = 1000
    x_end = min(x_start + dx, dims.X)
    y_end = min(y_start + dy, dims.Y)
    final_img = np.zeros((dims.Y, dims.X))

    for chunk in read_image_in_chunks(test_path, dx=dx, dy=dy, dz=0, dc=0, dt=1):
        chunk = chunk[0, :, :, :, :]
        chunk_projection = surface_projection_m(time_point=chunk, axes="CZXY", reference_channel=0, min_z=0, max_z=15, method="max_averages", bin_size=5)
        # chunk_projection = surface_projection_surf(chunk, "CZXY", 1, 0, 4, "max_averages", 5, chunk_number)
        final_img[y_start:y_end, x_start:x_end] = chunk_projection
        y_start += dy
        y_end = min(y_start + dy, dims.Y)
        if y_start >= dims.Y:
            x_start += dx
            x_end = min(x_start + dx, dims.X)
            y_start = 0
            y_end = min(y_start + dy, dims.Y)
            if x_start >= dims.X:
                break
    # col = 1
    # row = 0
    # chunk_number = 1
    # print(final_img.shape, 'x', final_img.shape[0], 'y', final_img.shape[1])
    # max_cols = int(np.ceil(dims.Y / dy))
    # max_rows = int(np.ceil(dims.X / dx))
    # print('max rows', max_rows, 'max columns', max_cols)
    # for chunk in read_image_in_chunks(test_path, dx=dx, dy=dy, dz=0, dc=0, dt=1):
    #     chunk = set_brightness(chunk, "TCZXY", clearExtreamPrecentage=1)
    #     chunk = chunk[0, :, :, :, :]
    #     chunk_projection = surface_projection_m(chunk, "CZXY", 1, 0, 4, "max_averages", 5)
    #     # chunk_projection = surface_projection_surf(chunk, "CZXY", 1, 0, 4, "max_averages", 5, chunk_number)
    #     if col % max_cols != 0:  # enters if it's not the last column and not the last row of thew grid
    #         print('enterd first if')
    #         x_start = row * dx
    #         x_stop = x_start + dx
    #         y_start = (col-1) * dy
    #         y_stop = y_start + dy
    #         print('x start/stop', x_start, x_stop, 'y start/stop', y_start, y_stop)
    #         final_img[x_start:x_stop, y_start:y_stop] = chunk_projection
    #         col += 1
    #     else:  # it's the final column or the final row
    #         if max(row, row + 1) % (max(1, max_rows-1)) == 0:  # enters if it's the last row
    #             break
    #         else:
    #             print('entered last column')
    #             # if col % (max_cols) == 0:  # enters if it's  the last column
    #             x_start = row * dx
    #             x_stop = x_start + dx
    #             y_start = (col-1) * dy
    #             y_stop = y_start + (dims.Y - y_start)
    #             print('x start/stop', x_start, x_stop, 'y start/stop', y_start, y_stop)
    #             final_img[x_start:x_stop, y_start:y_stop] = chunk_projection
    #             row += 1
    #             col = 1

        # print("chunk number:", chunk_number, 'original chunk shape', chunk.shape, 'chunk projection shape', chunk_projection.shape)
        print('___________________________________________________________________________')
        # chunk_number += 1
    final_img = set_brightness(final_img, "XY", clearExtreamPrecentage=3)
    print("end of loop")
    print('type', type(final_img))
    print('shape',final_img.shape)
    print(type(final_img[0][0]))
    min_val = final_img.min()
    max_val = final_img.max()
    print('min max val', min_val, max_val)
    print(final_img.dtype)
    save_tiff('no_method.tiff', final_img.reshape(final_img.shape), metadata={}, data_type="uint16")


