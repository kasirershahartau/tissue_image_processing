# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 17:01:39 2021

@author: Shahar Kasirer
"""
from basic_image_manipulations import *
from skimage.measure import block_reduce
from cv2 import resize

def surface_projection(time_point, axes, reference_channel, min_z, max_z,
                       method, bin_size):
    if axes.find("T") >= 0:
        time_point = time_point.reshape(time_point.shape[1:])
        image,_ = put_cannel_axis_first(time_point, axes[1:])
    else:
        image, _ = put_cannel_axis_first(time_point, axes)
    projection_channel = image[reference_channel]
    projection_channel = projection_channel[min_z:max_z]
    projection_channel = blur_image(projection_channel, (2,2,1))
    if method == "max_averages":
        score = block_reduce(projection_channel, (1,bin_size,bin_size), func=np.mean)
    elif method == "max_std":
        score = block_reduce(projection_channel, (1,bin_size,bin_size), func=np.var)
    else:
        raise "No such method %s"%method
    # chosen_z = np.repeat(np.repeat(min_z + np.argmax(score, axis=0), bin_size, axis=0), bin_size, axis=1)
    chosen_z = min_z + np.argmax(score, axis=0)
    z_size, y_size, x_size = image.shape[-3:]
    chosen_z = np.round(resize(chosen_z.astype('float32'), (x_size, y_size))).astype('int')
    if chosen_z.shape != image.shape[-2:]:
        chosen_z = chosen_z[:image.shape[-2],:image.shape[-1]]
    if axes.find("C") >= 0:
        projection = np.zeros((image.shape[0], image.shape[2], image.shape[3]))
        for channel_num in range(image.shape[0]):
            image[channel_num,:,:,:] = blur_image(image[channel_num,:,:,:], (5,5,2))
            reshaped = image[channel_num,:,:,:].reshape((z_size, x_size*y_size))
            projection[channel_num,:,:] = reshaped[chosen_z.flatten(), np.arange(x_size*y_size)].reshape((y_size, x_size))
    else:
        image = blur_image(image, (3,3,2))
        reshaped = image.reshape((z_size, x_size * y_size))
        projection = reshaped[chosen_z.flatten(), np.arange(x_size * y_size)].reshape((y_size, x_size))
    return projection
    
    


