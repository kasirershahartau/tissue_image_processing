# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 17:26:22 2021

@author: Shahar Kasirer

Small tests to debug image processing code
"""
from basic_image_manipulations import *
import os
from matplotlib import pyplot as plt

test_dir = "D:\\Anastasia\\Fixed_samples"
test_file = "Local Z Projection of 21_03_04_E17_utricle_Airyscan Processing_Stitch.tif"
test_path = os.path.join(test_dir, test_file)



def adjust_and_save(path):
    movie, axes, shape, metadata = read_tiff(path)
    data_type = movie.dtype
    movie, metadata = set_brightness(movie, axes, metadata, method='bestFit', clearExtreamPrecentage=1)
    
    save_tiff(path.replace('.tif', 'adjusted.tif'), movie, metadata=metadata, axes=axes, data_type=data_type)
    return

def loading_test_image():
     image, axes, shape, metadata = read_tiff(test_path)
     adjusted = set_channel_brightness(image[0], 65535)
     return adjusted

if __name__ == "__main__":
     pass
 
