# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 10:27:17 2021

@author: Shahar Kasirer, Anastasia Pergament

Basic tools for loading, saving and manipulating different kinds of images into numpy arrays
"""
from copy import deepcopy
import numpy as np
from scipy.stats import scoreatpercentile
from scipy.ndimage import gaussian_filter
from skimage.exposure import adjust_gamma
from tifffile import TiffFile, imwrite
from aicsimageio import AICSImage
from aicsimageio.readers import czi_reader, bioformats_reader
from aicsimageio.writers import ome_tiff_writer
from matplotlib import pyplot as plt
from cv2 import GaussianBlur as gaus
import cv2
import skimage.segmentation as skim
from skimage.filters import difference_of_gaussians, threshold_local
from scipy.fftpack import fftshift, fftn
from skimage.transform import resize
from skimage.registration import phase_cross_correlation

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


def read_whole_image(path, dims_order="TCZYX"):
    img = AICSImage(path)
    data = img.get_image_data(dims_order)
    return data, img.dims, img.metadata

def read_virtual_image(path, dims_order="TCZXY"):
    img = AICSImage(path)
    data = img.get_image_dask_data(dims_order)
    return data, img.dims, img.metadata

def read_part_of_image(path, x_range, y_range, z_range, c_range, t_range, dims_order="TCZXY"):
    img = AICSImage(path)
    default_dims_order = "TCZXY"
    data = img.get_image_dask_data(default_dims_order)
    data = data[t_range[0]:t_range[1],
                c_range[0]:c_range[1],
                z_range[0]:c_range[1],
                y_range[0]:y_range[1],
                x_range[0]:x_range[1]]
    data = data.compute()
    if default_dims_order != dims_order:
        permutation = [dims_order.index(default_dims_order[i]) for i in range(len(default_dims_order))]
        data = np.transpose(data, permutation)
    return data, img.dims, img.metadata

def get_image_dimensions(path, series=0):
    img = AICSImage(path, reader=bioformats_reader.BioformatsReader)
    img.set_scene(series)
    return img.dims

def get_image_metadata(path, series=0):
    img = AICSImage(path, reader=bioformats_reader.BioformatsReader)
    img.set_scene(series)
    return img.metadata

def read_image_in_chunks(path, series=0, dx=0, dy=0, dz=0, dc=0, dt=0, apply_function=None, output=None,
                         **apply_function_params):
    img = AICSImage(path,  reader=bioformats_reader.BioformatsReader, series=series, dask_tiles=True)
    default_dims_order = "TCZYX"
    if series == 0:  # There is an error with reading dask data for multi series image so we can only virtually read
        data = img.get_image_dask_data(default_dims_order)
    else:
        data = img.data
    max_x = img.dims.X
    max_y = img.dims.Y
    max_z = img.dims.Z
    max_t = img.dims.T
    max_c = img.dims.C
    if dx == 0:
        dx = max_x
    if dy == 0:
        dy = max_y
    if dz == 0:
        dz = max_z
    if dc == 0:
        dc = max_c
    if dt == 0:
        dt = max_t
    t = 0
    c = 0
    z = 0
    x = 0
    y = 0
    while t < max_t:
        while c < max_c:
            while z < max_z:
                while y < max_y:
                    while x < max_x:
                        chunk = data[t:min(t+dt, max_t),
                                     c:min(c+dc, max_c),
                                     z:min(z+dz, max_z),
                                     y:min(y+dy, max_y),
                                     x:min(x+dx, max_x)]
                        if series == 0:
                            chunk = chunk.compute()
                        if apply_function is None:
                                yield chunk
                        else:
                            result = apply_function(chunk, **apply_function_params)
                            if output is not None:
                                deflate = False
                                if not isinstance(result, tuple):
                                    deflate = True
                                    result = [result]
                                    output = [output]
                                for i in range(len(result)):
                                    out_t, out_c, out_z, out_y, out_x = output[i].shape
                                    output[i][min(t, out_t):min(t+dt, max_t, out_t),
                                         min(c, out_c):min(c+dc, max_c, out_c),
                                         min(z, out_z):min(z+dz, max_z, out_z),
                                         min(y, out_y):min(y+dy, max_y, out_y),
                                         min(x, out_x):min(x+dx, max_x, out_x)] = result[i].reshape((min(t+dt, max_t, out_t)-min(t, out_t),
                                                                                                  min(c+dc, max_c, out_c)-min(c, out_c),
                                                                                                  min(z+dz, max_z, out_z)-min(z, out_z),
                                                                                                  min(y+dy, max_y, out_y)-min(y, out_y),
                                                                                                  min(x+dx, max_x, out_x)-min(x, out_x)))
                                if deflate:
                                    result = result[0]
                                yield result
                        x+=dx
                    x=0
                    y+=dy
                y=0
                z+=dz
            z=0    
            c+=dc
        c=0    
        t+=dt
    return


def save_tiff(path, image, metadata=None, axes="", data_type=""):
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
    if data_type:
        if image.dtype != data_type and (data_type == 'uint8' or data_type == 'uint16'):
            max_possible_val = UINT8_MAXVAL if data_type == 'uint8' else UINT16_MAXVAL
            image = np.round((image/np.max(image))*max_possible_val).astype(data_type)
    writer = ome_tiff_writer.OmeTiffWriter
    writer.save(image, path, dim_order=axes, ome_xml=metadata)
    return

def create_imageJ_metadata(img_shape):
    frames = img_shape[0]
    channels = img_shape[1]
    z_stacks = img_shape[3] if len(img_shape) > 4 else 1
    x_size = img_shape[-1]
    y_size = img_shape[-2]
    meta = {"ImageJ": '1.53f', "images": ''}

def put_channel_axis_first(image, axes):
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

def set_brightness(image, axes, metadata={}, method='bestFit', clearExtreamPrecentage=1, minVal=0, maxVal=0):
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
    if maxVal:
        max_possible_val = maxVal
    else:
        max_possible_val = 255 if data_type == 'uint8' else 65535 if data_type == 'uint16' else 1
    adjusted = np.copy(image).astype('double')
    channel_axis = axes.find("C")
    minimum_pixel_val = max(minVal, 0)
    if metadata:
        if 'min' in metadata:
            minimum_pixel_val = metadata['min']
    if channel_axis >= 0:  # channel axis exists
        adjusted, desired_order = put_channel_axis_first(adjusted, axes)
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
    else:
        new_minimum = minimum_pixel_val
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
        adjusted, order = put_channel_axis_first(adjusted, axes)
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
        

      
def blur_image(image, std):
    """
   Blurs image using GaussianBlur 
    Parameters
    ----------
    image : matrix
        Object for blurring
    std : Tuple with size as the number of dimensions of the input image.
        Standard deviation for each dimension. Used for the intensity of the blur. Higher value outputs higher blur.

    Returns
    -------
    filtered : matrix
        Blurred image according to set parameters.

    """
    filtered = gaussian_filter(image, std, mode='nearest')
    return filtered


def band_pass_filter(image, lowsigma, highsigma):
    """
    
    Applies the band pass as a range of signals to accentuate in image.
    Parameters
    ----------
    image : matrix
        Object to put through the filter.
    lowsigma : Number
        Low end of wanted frequencies.
    highsigma : Number
        High end of wanted frequencies.

    Returns
    -------
    adjust: matrix
        Filtered image according to parameters.

    """
    
    adjust = difference_of_gaussians(image, lowsigma, highsigma)
    return adjust


def watershed_segmentation(image, imgthresh, stdeviation):
    """
    
    Applies watershed segmentation to create "skeleton" version of image for analysis
    Parameters
    ----------
    image : matrix
        Object to be segmented.
    imgthresh : number
        Parameter for segmenting the frequencies of the image. Threshold for 
        differing images to use. 
    stdeviation : number
        Standard deviation for applying filter.

    Returns
    -------
    skeleton: matrix
        Filtered image according to parameters.
    labelled: matrix
        Segmented cells from image

    """
    seg = np.copy(image)
    seg[seg < imgthresh] = 0
    blurred = blur_image(seg, stdeviation) #bigger std takes away more lines, bigger kern adds lines, used to blur the image
    labelled = skim.watershed(blurred, watershed_line=True) #used to list all the cells
    return labelled


def watershed_segmentation(image, imgthresh, stdeviation, blocksize):
    """

    Applies watershed segmentation to create "skeleton" version of image for analysis
    Parameters
    ----------
    image : matrix
        Object to be segmented.
    imgthresh : number between 0-1
        thresholding every pixl that is smaller than imgthresh*maximum intensity at each local environment
    stdeviation : number
        Standard deviation for applying filter.

    Returns
    -------
    skeleton: matrix
        Filtered image according to parameters.
    labelled: matrix
        Segmented cells from image

    """
    seg = np.copy(image)
    def local_thresh_helper(flatten_array):
        return imgthresh*np.max(flatten_array)
    if blocksize % 2 == 0:
        blocksize += 1
    threshold = threshold_local(seg, block_size=blocksize, method='generic', param=local_thresh_helper)
    seg[seg < threshold] = 0
    blurred = blur_image(seg, stdeviation)  # bigger std takes away more lines, bigger kern adds lines, used to blur the image
    labelled = skim.watershed(blurred, watershed_line=True)  # used to list all the cells
    return labelled

def concatenate_time_points(files):
    imgs = []
    for file in files:
        img = np.load(file)
        imgs.append(img)
        if img.shape[1:] != imgs[0].shape[1:]:
            imgs[-1] = resize(img, (img.shape[0], imgs[0].shape[1:]))
    return np.concatenate(imgs, axis=0)

def calculate_drift(first_image, second_image, sub_pixel_precision=True):
    """
    Calculating the global 2D drift between 2 images (usually 2 time-lapse frames).
    @param first_image: The reference image (can be multi channels image)
    @param second_image: The drifted image. Same size as first_image.
    @param sub_pixel_precision: If True (default), the shift would be given to sub-pixel precision (floating numbers),
    else it would be a whole number.
    @returns horizontal shift, vertical shift. Shift is what needed to be added to the second image to get the first
    image. The positive directions are down and right.
    """
    if sub_pixel_precision:
        shift, error, diffphase = phase_cross_correlation(first_image, second_image, upsample_factor=100)
    else:
        shift, error, diffphase = phase_cross_correlation(first_image, second_image)
    return shift[-2:]

