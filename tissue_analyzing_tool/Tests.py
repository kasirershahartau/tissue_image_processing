# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 17:26:22 2021

@author: Shahar Kasirer

Small tests to debug image processing code
"""
from basic_image_manipulations import *
import os
from matplotlib import pyplot as plt

test_dir = "C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\experimental_results\\fixed\\Vestibule\\zo_myo_dapi\\E17.5"
test_file = "21-04-13_E17.5_utricle2_zo1_atho1_dapi_Airyscan Processing.czi"

test_path = os.path.join(test_dir, test_file)



def adjust_and_save(path):
    movie, axes, shape, metadata = read_tiff(path)
    data_type = movie.dtype
    movie, metadata = set_brightness(movie, axes, metadata, method='bestFit', clearExtreamPrecentage=1)
    
    save_tiff(path.replace('.tif', 'adjusted.tif'), movie, metadata=metadata, axes=axes, data_type=data_type)
    return



if __name__ == "__main__": 
    i=0
    for chunk in read_image_in_chunks(test_path, dx = 1000, dy = 1000, dz = 0, dc = 0, dt= 1):
        chunk = set_brightness(chunk, "TCZXY", clearExtreamPrecentage=0)
        save_tiff(os.path.join(test_dir, "chunk%d.tif"%i), (chunk*255).astype(np.uint8))
        i+=1