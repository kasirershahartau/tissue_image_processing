# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 17:01:39 2021

@author: Shahar Kasirer
"""
import os.path
import pickle
from ast import literal_eval
import numpy as np
from optparse import OptionParser
from basic_image_manipulations import *
from skimage.measure import block_reduce
from skimage.transform import resize

def time_point_surface_projection(time_point, axes, reference_channel, min_z=0, max_z=0,
                       method='max_averages', bin_size=1, airyscan=True, z_map=False):
    if axes.find("T") >= 0:
        time_point = time_point.reshape(time_point.shape[1:])
        image,_ = put_channel_axis_first(time_point, axes[1:])
    else:
        image, _ = put_channel_axis_first(time_point, axes)
    image = image.astype('float32')
    if airyscan:
        image -= 10000
        image[image < 0] = 0
    projection_channel = np.copy(image[reference_channel])
    projection_channel = blur_image(projection_channel, (0.5, 1, 1))
    if max_z > 0:
        projection_channel = projection_channel[min_z:max_z]
    if bin_size > 1:
        if method == "max_averages":
            score = block_reduce(blur_image(projection_channel, (0.5, 30, 30)), (1,bin_size,bin_size), func=np.mean)
        elif method == "max_std":
            score = block_reduce(projection_channel, (1,bin_size,bin_size), func=np.var)
        else:
            raise "No such method %s"%method
    else:
        score = blur_image(projection_channel, (0.5, 30, 30))
    chosen_z = min_z + np.argmax(score, axis=0)
    z_size, y_size, x_size = image.shape[-3:]
    if chosen_z.shape != (y_size, x_size):
        chosen_z = np.round(resize(chosen_z.astype('float32'), (y_size, x_size))).astype('int')
    projection_mask = np.zeros((z_size, y_size*x_size))
    projection_mask[chosen_z.flatten(), np.arange(x_size*y_size)] = 1
    projection_mask = blur_image(projection_mask.reshape((z_size, y_size, x_size)), (1, 2, 2))
    if axes.find("C") >= 0:
        channels = image.shape[0]
        projection = np.zeros((channels, y_size, x_size))
        for channel_num in range(channels):
            projection[channel_num,:,:] = np.max(image[channel_num]*projection_mask, axis=0)
    else:
        projection = np.max(image*projection_mask, axis=0)
    if z_map:
        return projection, chosen_z
    else:
        return projection
    
def movie_surface_projection(files, reference_channel, position_final_movie, initial_positions_number, output_dir):
    """
    @param files: list of movie czi files in order
    @param reference_channel: Index of channel that will be used for projection
    @param position_final_movie: list of final file number in which each position still exists. For example if the
     sample that was originally in position 1 was filmed up to movie 3 and the sample that was originally in position 2
     was filmed up to movie 4 we set position_final_movie=(3,2)
    @param initial_positions_number: number of positions in the first file.
    @param output_dir: path for output directory
    """
    # Initializing output with X,Y dimensions according with the first file
    dims = get_image_dimensions(files[0])
    projections = [None] * initial_positions_number
    z_maps = [None] * initial_positions_number
    for i in range(len(projections)):
        projections[i] = np.empty((0,dims.C, dims.Y, dims.X))
        z_maps[i] = np.empty((0, dims.Y, dims.X))
    positions = list(range(initial_positions_number))
    # Projecting
    for file_num, file in enumerate(files):
        remove_positions = []
        for position_num, position in enumerate(positions):
            print("Projecting position %d, movie %d" % (position + 1, file_num + 1))
            dims = get_image_dimensions(file)
            current_projection = np.zeros((dims.T, dims.C, 1, dims.Y, dims.X))
            current_zmap = np.zeros((dims.T, 1, 1, dims.Y, dims.X))
            projector = read_image_in_chunks(file, series=position_num, dt=1,
                                             apply_function=time_point_surface_projection,
                                             output=[current_projection, current_zmap], axes='TCZYX',
                                             reference_channel=reference_channel, z_map=True)
            for time_point_index, time_point in enumerate(projector):
                print("Projecting timepoint %d" % (time_point_index + 1))
            current_projection = current_projection.reshape((dims.T, dims.C, dims.Y, dims.X))
            if projections[position].shape[1:] != (dims.C, dims.Y, dims.X):
                current_projection = resize(current_projection, (dims.T,)+projections[position].shape[1:])
            current_zmap = current_zmap.reshape((dims.T, dims.Y, dims.X))
            if z_maps[position].shape[1:] != (dims.Y, dims.X):
                current_zmap = resize(current_zmap, (dims.T,)+z_maps[position].shape[1:])
            projections[position] = np.concatenate((projections[position], current_projection), axis=0)
            z_maps[position] = np.concatenate((z_maps[position], current_zmap), axis=0)
            if position_final_movie[position] == file_num + 1:
                remove_positions.append(position)
        for to_delete in remove_positions:
            positions.remove(to_delete)
    # Updating meta data
    meta = []
    for i in range(len(projections)):
        former_metadata = get_image_metadata(files[0], series=i)
        new_metadata = update_projection_metadata(former_metadata, projections[i].shape[0], series=i)
        meta.append(new_metadata)
    # Saving projections
    for i,proj in enumerate(projections):
        save_tiff(os.path.join(output_dir, "position%d.tif" %(i+1)), proj, metadata=meta[i], axes="TCYX",data_type="uint16")
    for i, z_map in enumerate(z_maps):
        np.save(os.path.join(output_dir, "zmap%d.npy" %(i+1)), z_map)
    # Saving stage location (to correct for stage movement between movies)
    save_stage_positions(files, position_final_movie, initial_positions_number, output_dir)

def save_stage_positions(files, position_final_movie, initial_positions_number, output_dir):
    positions = list(range(initial_positions_number))
    meta = get_image_metadata(files[0])
    stage_pos = [{"x": [meta.images[i].stage_label.x]*meta.images[i].pixels.size_t,
                  "y": [meta.images[i].stage_label.y]*meta.images[i].pixels.size_t,
                  "z": [meta.images[i].stage_label.z]*meta.images[i].pixels.size_t,
                  "x_unit":meta.images[i].stage_label.x_unit,
                  "y_unit":meta.images[i].stage_label.y_unit,
                  "z_unit":meta.images[i].stage_label.z_unit,
                  "physical_size_x":meta.images[i].pixels.physical_size_x,
                  "physical_size_y":meta.images[i].pixels.physical_size_y,
                  "physical_size_z":meta.images[i].pixels.physical_size_z} for i in range(initial_positions_number)]
    for file_index in range(1,len(files)):
        meta = get_image_metadata(files[file_index])
        remove_positions = []
        for position_index, position in enumerate(positions):
            stage_pos[position]["x"].extend(
                [meta.images[position_index].stage_label.x]*meta.images[position_index].pixels.size_t)
            stage_pos[position]["y"].extend(
                [meta.images[position_index].stage_label.y] * meta.images[position_index].pixels.size_t)
            stage_pos[position]["z"].extend(
                [meta.images[position_index].stage_label.z] * meta.images[position_index].pixels.size_t)
            if position_final_movie[position] == file_index + 1:
                remove_positions.append(position)
        for to_delete in remove_positions:
            positions.remove(to_delete)
    for i in range(initial_positions_number):
        out_path = os.path.join(output_dir, "stage_locations_position%d.pkl" %(i + 1))
        with open(out_path, 'wb') as f:
            pickle.dump(stage_pos[i], f)


def update_projection_metadata(metadata, frames_number, series=0):
    metadata.images = [metadata.images[series]]
    metadata.images[0].name = 'position%d' % series
    metadata.images[0].pixels.dimension_order = 'XYCTZ'
    metadata.images[0].pixels.size_z = 1
    metadata.images[0].pixels.size_t = frames_number
    metadata.images[0].pixels.type = 'uint16'
    metadata.images[0].pixels.planes = metadata.images[0].pixels.planes[:metadata.images[0].pixels.size_c]
    return metadata

def getOptions():
    """
    Collecting command line parameters
    """
    parser = OptionParser(usage="usage: %prog [options]")
    parser.add_option("-i", "--input", dest="input",
                      help="Input directory with movies m1, m2, m3, ... [default: current directory]", default="")
    parser.add_option("-o", "--output", dest="output", help="Output directory [default: same as input directory",
                      default="")
    parser.add_option("-f", "--position-final_movie", dest="position_final_movie",
                      help="Final movie for each sample in the order of initial movie positions [default: all finish in"
                           " last movie]", default="")
    parser.add_option("-n", "--position-number", dest="position_number",
                      help="Number of initial positions [default: 1]", type=int, default=1)
    parser.add_option("-m", "--movie-number", dest="movie_number",
                      help="Number of movies [default: 1]", type=int, default=1)
    parser.add_option("-r", "--reference_channel", dest="reference_channel",
                      help="Reference of the channel that will be used for projection [default: 1]",
                      type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    options, args = getOptions()
    input_dir = options.input
    if not input_dir:
        input_dir = os.getcwd()
    output_dir = options.output
    if not output_dir:
        output_dir = input_dir
    position_number = options.position_number
    movie_number = options.movie_number
    position_final_movie = options.position_final_movie
    if not position_final_movie:
        position_final_movie = [movie_number]*position_number
    else:
        position_final_movie = list(literal_eval(position_final_movie))
    files = [os.path.join(input_dir,"m%d.czi" %(i + 1)) for i in range(movie_number)]
    reference_channel = options.reference_channel
    movie_surface_projection(files, reference_channel, position_final_movie, position_number, output_dir)
    exit()



