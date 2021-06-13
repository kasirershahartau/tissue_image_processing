# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 17:26:22 2021

@author: Shahar Kasirer

Small tests to debug image processing code
"""
from basic_image_manipulations import *
import os
from matplotlib import pyplot as plt

test_dir = "C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\experimental_results\\processed_data"
test_file = "21_01_06_Macula_p0_Airyscan Processing_Stitch_LocalProjection_0.tif"

test_path = os.path.join(test_dir, test_file)



def adjust_and_save(path):
    movie, axes, shape, metadata = read_tiff(path)
    data_type = movie.dtype
    movie, metadata = set_brightness(movie, axes, metadata, method='bestFit', clearExtreamPrecentage=1)
    
    save_tiff(path.replace('.tif', 'adjusted.tif'), movie, metadata=metadata, axes=axes, data_type=data_type)
    return

if __name__ == "__main__":
    adjust_and_save(test_path)