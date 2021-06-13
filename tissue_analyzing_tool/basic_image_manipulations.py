# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 10:27:17 2021

@author: Shahar Kasirer

Basic tools for loading, saving and manipulating different kinds of images into numpy arrays
"""
from copy import deepcopy
import numpy as np
from scipy.stats import scoreatpercentile
from skimage.exposure import adjust_gamma
from tifffile import TiffFile, imwrite

####### Constants ##############
UINT8_MAXVAL = 255
UINT16_MAXVAL = 65535


def read_tiff(path):
    """
    Reading the given tiff file. 
     Parameters
    ----------
    path - string
       output path
    Returns
    -------
    tuple, (image, axes, shape, metadata)
    image - numpy array of image
    axes - a string specifying axis order (i.e. 'TXY' means that the first
                                           axis is time, second is x and third
                                           is y)
    shape - Shape of image, same order as axes (i.e. if axes='TXY' and 
                                                shape=(3,5,5) we have 3
                                                timepoints, 5X5 pixels for
                                                each). 
    """
    with TiffFile(path) as tif:
        image = tif.asarray()
        axes = tif.series[0].axes
        metadata = tif.imagej_metadata
    return image, axes, image.shape, metadata    


def save_tiff(path, image, metadata={}, axes="", data_type=""):
    """
    Saving the given image as a tif file.
    Parameters
    ----------
    path : string
        Input path
    image : numpy array 
        Image data
    metadata : dictionary, optional.
        Image meta-data. Default is an empty dictionary.
    axes : String, optional.
        a string specifying axis order (i.e. 'TXY' means that the first
                                        axis is time, second is x and third
                                        is y). Default is "".
    data_type : String, optional.
        saved image data type (uint8, uint16 or float). Default is not to
        change the input image data type.
        
        
    """
    if axes:
        metadata['axes'] = axes    
    if data_type:
        if image.dtype != data_type and (data_type == 'uint8' or data_type == 'uint16'):
            max_possible_val = UINT8_MAXVAL if data_type == 'uint8' else UINT16_MAXVAL
            image = np.round((image/np.max(image))*max_possible_val).astype(data_type)
    imwrite(path, image, imagej=True, metadata=metadata)
    return

def put_cannel_axis_first(image, axes):
    """
    Transposing image so that channels axis would be 0. The order would be
    "CTZXY" in the case of a 3D movie.
    Parameters
    ----------
    image : numpy array.
        Image data.
    axes : string.
         A string specifying axis order (i.e. 'TXY' means that the first
                                           axis is time, second is x and third
                                           is y)
    Returns: image, order.
    -------
    numpy array, tuple.
        The transposed image and the order in which it had been transposed.

    """
    channel_axis = axes.find("C")
    if channel_axis > 0:
        time_axis = axes.find("T")
        x_axis = axes.find("X")
        y_axis = axes.find("Y")
        z_axis = axes.find("Z")
        desired_order =  (x_axis, y_axis)
        if z_axis >= 0:  # z axis exists
            desired_order = (z_axis,) + desired_order
        if time_axis>=0: # time axis exists
            desired_order = (time_axis,) + desired_order
        desired_order = (channel_axis,) + desired_order
        return np.transpose(image,axes=desired_order), desired_order
    else:
        return image, tuple(np.arange(len(axes)))

def set_brightness(image, axes, metadata={}, method='bestFit', clearExtreamPrecentage=1):
    """
    Adjusteing the brightness of each pixel in the given image/movie and
    transforms pixels value into floats in the range [0,1]. Applied on each
    channel seperately.
    Parameters
    ----------
    image : numpy array.
        image/movie data
    axes : string.
        A string specifying axis order (i.e. 'TXY' means that the first
                                           axis is time, second is x and third
                                           is y)
    metadata: dictionary, optional.
        If given, an adjusted copy of metadata would be returned.
    method : string, optional
        Method for adjusting brightnss. 'minMax' - Linearly transforms the pixel
        values such that the minimum intensity would be 0 and the maximum 1.
        'bestFit': Finds the best linear transformation according to intensity
        histogram. The default is 'bestFit'.
    clearExtreamPrecentage : integer between 0-100, optional.
        Saturates the top and buttom intensity pixels according to the given
        precentage. The default is 0 (does nothing).
    Returns
    -------
    adjusted image : numpy array.
        image with adjusted brightness with pixels intensity
        in the range [0,1].
    If metadata is not empty, a tuple would be returned: (adjusted image,
                                                          adjusted metadata).

    """
    data_type = image.dtype
    max_possible_val = 255 if data_type == 'uint8' else 65535 if data_type == 'uint16' else 1
    adjusted = np.copy(image).astype('double')
    channel_axis = axes.find("C")
    minimum_pixel_val = 0
    if metadata:
        if 'min' in metadata:
            minimum_pixel_val = metadata['min']
    if channel_axis >= 0:  # channel axis exists
        adjusted, desired_order = put_cannel_axis_first(adjusted, axes)
        number_of_channels = adjusted.shape[0]
        for channel in range(number_of_channels):
            current = adjusted[channel]
            current = set_channel_brightness(current, max_possible_val,
                                             method, clearExtreamPrecentage,
                                             minimum_pixel_val)   
            adjusted[channel] = current
        adjusted = np.transpose(adjusted, axes=(np.argsort(desired_order))) #  Reverting changes
    else:
        adjusted = set_channel_brightness(adjusted, max_possible_val,
                                      method, clearExtreamPrecentage,
                                      minimum_pixel_val)   
    if metadata:
        adjusted_metadata = deepcopy(metadata)
        if 'min' in adjusted_metadata:
            adjusted_metadata['min'] = 0
        if 'max' in adjusted_metadata:
            adjusted_metadata['max'] = max_possible_val
        if 'Ranges' in adjusted_metadata:
            adjusted_metadata['Ranges'] = (0, max_possible_val) *\
            int(len(adjusted_metadata['Ranges'])//2)
        return adjusted, adjusted_metadata   
    else:
        return adjusted

def set_channel_brightness(image, max_possible_val, method='bestFit',
                           clearExtreamPrecentage=1, minimum_pixel_val=0):
    """
    Adjusteing the brightness of each pixel in the given channel and
    transforms pixels value into floats in the range [0,1].
    Parameters
    ----------
    image : numpy array.
        Channel data
    max_possible_vel : int.
        Maximum possible pixel value (e.g. 255 for 8 bit uint image)
    method : string, optional
        Method for adjusting brightnss. 'minMax' - Linearly transforms the pixel
        values such that the minimum intensity would be 0 and the maximum 1.
        'bestFit': Finds the best linear transformation according to intensity
        histogram. The default is 'bestFit'.
    clearExtreamPrecentage : integer between 0-100, optional.
        Saturates the top and buttom intensity pixels according to the given
        precentage. The default is 0 (does nothing).
    minimum_pixel_val : int.
        The actual minimum pixel value with significant (if known). Default is
        0 (unknown).
    Returns
    -------
    adjusted : numpy array.
        channel with adjusted brightness with pixels intensity
        in the range [0,1].

    """
    
    if clearExtreamPrecentage > 0:
        new_maximum = scoreatpercentile(image, 100-clearExtreamPrecentage)
        new_minimum = scoreatpercentile(image, clearExtreamPrecentage)
        if minimum_pixel_val > 0:
            new_minimum = max(new_minimum, minimum_pixel_val)
        image[image > new_maximum] = new_maximum
    if method == 'minMax' or method == 'bestFit':
        image = image - new_minimum
        image = image/np.max(image)
        image = image + 1/max_possible_val
        image[image < 0] = 0
    if method == 'bestFit':
        image = adjust_gamma(image)
    return image    

def binary_image(image, axes, thresholds):
    adjusted = np.copy(image)
    channel_axis = axes.find("C")
    if channel_axis > 0:
        adjusted, order = put_cannel_axis_first(adjusted, axes)
        number_of_channels = adjusted.shape[0]
        for channel in range(number_of_channels):
            current = adjusted[channel]
            threshold = thresholds[channel] if hasattr(thresholds, "__len__")\
                else thresholds    
            current[current > threshold] = 1
            current[current < threshold] = 0
            adjusted[channel] = current
        adjusted = np.transpose(adjusted, axes=(np.argsort(order))) #  Reverting changes 
    else:
        threshold = thresholds[0] if hasattr(thresholds, "__len__")\
                    else thresholds
        adjusted[adjusted > threshold] = 1
        adjusted[adjusted < threshold] = 0
    return adjusted 
        
        
