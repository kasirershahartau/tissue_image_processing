# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 17:01:39 2021

@author: Shahar Kasirer
"""
from basic_image_manipulations import *
from skimage.measure import block_reduce
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np


def surface_projection_m(time_point, axes, reference_channel, min_z, max_z, method, bin_size):
    image, _ = put_cannel_axis_first(time_point, axes)
    projection_channel = image[reference_channel]
    projection_channel = projection_channel[min_z:max_z]
    projection_channel = blur_image(projection_channel, (5, 5, 3))
    min_val = projection_channel.min()
    max_val = projection_channel.max()
    print('min max val', min_val, max_val)
    if method == "max_averages":
        score = block_reduce(projection_channel, (1, bin_size, bin_size), func=np.mean)
    elif method == "max_std":
        score = block_reduce(projection_channel, (1, bin_size, bin_size), func=np.var)
    else:
        raise "No such method %s" % method
    ex_score = expend_score(score, bin_size)
    fixed_score = shapes_are_same(ex_score, projection_channel)
    best_z = np.argmax(fixed_score, axis=2)
    projection = choose_z(projection_channel, best_z)
    # name = str(chunk_number) + '.tiff'
    # image = Image.fromarray(projection, mode='F')
    # image.save(name, 'TIFF')
    return projection

def choose_z(projection_channel, best_z):
    z, x, y = projection_channel.shape
    projection_channel.reshape((z, (x*y)))
    result = np.choose(best_z, projection_channel)
    return result.reshape((x,y))


def shapes_are_same(score, projection_channel):
    z, x, y = projection_channel.shape
    fixed_score = score[:x, :y, :z]
    return fixed_score


def rearrange_axis(expended_score):
    score_z, score_x, score_y = sorted(expended_score.shape, reverse=False)
    score_i_list = [score_z, score_x, score_y]
    z = min(score_i_list)
    score_i_list.remove(z)
    last = min(score_i_list)
    mid = max(score_i_list)
    re_expended_score = expended_score.reshape((z, mid, last))
    return re_expended_score


def find_best_pixels(projection_channel, score):
    img = np.zeros(shape=(projection_channel.shape[1], projection_channel.shape[2]))
    for x in range(projection_channel.shape[1]):
        for y in range(projection_channel.shape[2]):
            z = find_best_z(projection_channel, x, y, score)
            pixel = projection_channel[z][x][y]
            img[x][y] = pixel
    return img

#check this function
def find_best_z(projection_channel, x, y, score):
    best_z = 0
    max_score = score[x][y].max()
    for z in range(projection_channel.shape[0]):
        if projection_channel[z][x][y] > max_score:
            max_score = projection_channel[z][x][y]
            best_z = z
    return best_z


def expend_score(score, bin_size):
    rows_list = []
    list_of_arr = []
    arr_2d_list = []

    for z in range(score.shape[0]):
        for x in range(score.shape[1]):
            for y in range(score.shape[2]):
                arr = np.repeat(score[z][x][y], bin_size)
                list_of_arr.append(arr)
            arr_of_list = np.array(list_of_arr)
            row_arr = np.concatenate(arr_of_list, axis=None)
            for i in range(bin_size):
                rows_list.append(row_arr)
            list_of_arr = []
        arr_2d = np.vstack(rows_list)
        arr_2d_list.append(arr_2d)
        rows_list = []
    new_score = np.dstack(np.array(arr_2d_list))
    return new_score


def make_2d_score_metrix(expended_score):
    w, h = (expended_score.shape[1], expended_score.shape[2])
    score_2d = np.zeros(shape=(w, h))
    for z in range(expended_score.shape[0]):
        max_score = expended_score[z].max()
        for x in range(expended_score.shape[1]):
            for y in range(expended_score.shape[2]):
                if expended_score[z][x][y] >= max_score:
                    max_score = expended_score[z][x][y]
                score_2d[x][y] = max_score
    return score_2d



