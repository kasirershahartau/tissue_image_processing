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
                       method='max_averages', bin_size=1, airyscan=True, z_map=False, atoh_shift=0,
                                  build_manifold=False):

    if axes.find("T") >= 0:
        time_point = time_point.reshape(time_point.shape[1:])
        image,_ = put_channel_axis_first(time_point, axes[1:])
    else:
        image, _ = put_channel_axis_first(time_point, axes)
    image = image.astype('float32')
    if airyscan:
        image -= 10000
        image[image < 0] = 0
    if max_z > 0:
        image = image[:,min_z:max_z,:,:]
    projection_channel = np.copy(image[reference_channel])
    percentile95 = np.percentile(projection_channel[projection_channel > 0], 95)
    projection_channel[projection_channel > percentile95] = percentile95
    projection_channel = blur_image(projection_channel, (0.5, 1, 1))
    z_size, y_size, x_size = image.shape[-3:]
    if bin_size > 1:
        if method == "max_averages":
            score = block_reduce(blur_image(projection_channel, (0.5, 30, 30)), (1,bin_size,bin_size), func=np.mean)
        elif method == "max_std":
            score = block_reduce(projection_channel, (1,bin_size,bin_size), func=np.var)
        elif method == "multi_channel":
            atoh_channel = np.copy(image[(reference_channel+1) % image.shape[0]])
            atoh_percentile95 = np.percentile(atoh_channel, 95)
            atoh_channel[atoh_channel > atoh_percentile95] = atoh_percentile95
            atoh_channel = blur_image(atoh_channel, (0.5, 1, 1))
            zo_score = block_reduce(projection_channel, (1,bin_size,bin_size), func=np.var)
            atoh_score = block_reduce(blur_image(atoh_channel, (0.5, 30, 30)), (1,bin_size,bin_size), func=np.mean)
            score = atoh_score*zo_score
        else:
            raise "No such method %s"%method
    else:
        score = blur_image(projection_channel, (0.5, 30, 30))
    if build_manifold:
        chosen_z = build_continues_manifold(score)
    else:
        if score.shape[1:] != (y_size, x_size):
            score = resize(score.astype('float32'), (z_size, y_size, x_size))
        chosen_z = min_z + np.argmax(score, axis=0)
    chosen_z_atoh = np.copy(chosen_z) if atoh_shift == 0 else np.clip(chosen_z + atoh_shift, 0, score.shape[0])
    if chosen_z.shape != (y_size, x_size):
        chosen_z = np.round(resize(chosen_z.astype('float32'), (y_size, x_size))).astype('int')
        chosen_z_atoh = np.round(resize(chosen_z_atoh.astype('float32'), (y_size, x_size))).astype('int')
    projection_mask = np.zeros((z_size, y_size*x_size))
    projection_mask_atoh = np.zeros((z_size, y_size*x_size))
    projection_mask[chosen_z.flatten(), np.arange(x_size*y_size)] = 1
    projection_mask_atoh[chosen_z_atoh.flatten(), np.arange(x_size * y_size)] = 1
    projection_mask = blur_image(projection_mask.reshape((z_size, y_size, x_size)), (1, 2, 2))
    projection_mask_atoh = blur_image(projection_mask_atoh.reshape((z_size, y_size, x_size)), (1, 2, 2))
    if axes.find("C") >= 0:
        channels = image.shape[0]
        projection = np.zeros((channels, y_size, x_size))
        for channel_num in range(channels):
            if channel_num == reference_channel:
                projection[channel_num,:,:] = np.max(image[channel_num]*projection_mask, axis=0)
            else:
                projection[channel_num, :, :] = np.max(image[channel_num] * projection_mask_atoh, axis=0)
    else:
        projection = np.max(image*projection_mask, axis=0)
    if z_map:
        return projection, chosen_z
    else:
        return projection

def build_continues_manifold(score):
    chosen_z = -1 * np.ones(score.shape[1:]).astype(int)
    max_row = score.shape[1]
    max_col = score.shape[2]
    max_plane = score.shape[0]
    start_plane, start_row, start_col = np.unravel_index(np.argmax(score), score.shape)
    chosen_z[start_row, start_col] = start_plane
    distance_from_edge = np.max(np.abs(np.array([start_col, start_row, start_col, start_row]) - np.array([0,0,max_col-1, max_row-1])))
    distance = 1
    while distance <= distance_from_edge:
        # right edge lower half
        col = start_col + distance
        if col < max_col:
            for row in range(start_row, start_row + distance + 1):
                if row < max_row:
                    chosen_z[row, col] = find_pixel_plane(score, chosen_z, row, col, max_row, max_col, max_plane)
        # bottom edge
        row = start_row + distance
        if row < max_row:
            for col in range(start_col + distance - 1, start_col - distance - 1, -1):
                if 0 <= col < max_col:
                    chosen_z[row, col] = find_pixel_plane(score, chosen_z, row, col, max_row, max_col, max_plane)
        # left edge
        col = start_col - distance
        if col >= 0:
            for row in range(start_row + distance - 1, start_row - distance - 1, -1):
                if 0 <= row < max_row:
                    chosen_z[row, col] = find_pixel_plane(score, chosen_z, row, col, max_row, max_col, max_plane)
        # upper edge
        row = start_row - distance
        if row >= 0:
            for col in range(start_col - distance + 1, start_col + distance + 1):
                if 0 <= col < max_col:
                    chosen_z[row, col] = find_pixel_plane(score, chosen_z, row, col, max_row, max_col, max_plane)
        # right edge upper half
        col = start_col + distance
        if col < max_col:
            for row in range(start_row - distance + 1, start_row):
                if row >= 0:
                    chosen_z[row, col] = find_pixel_plane(score, chosen_z, row, col, max_row, max_col, max_plane)
        distance += 1
    return chosen_z

def find_pixel_plane(score, chozen_z, pixel_row, pixel_col, max_row, max_col, max_plane):
    neighbor1_plane = None
    neighbor2_plane = None
    if pixel_row >= 0:
        plane = chozen_z[pixel_row - 1, pixel_col]
        if plane >= 0:
            neighbor1_plane = plane
    if pixel_row < max_row - 1:
        plane = chozen_z[pixel_row + 1, pixel_col]
        if plane >= 0:
            if neighbor1_plane is None:
                neighbor1_plane = plane
            else:
                neighbor2_plane = plane
    if neighbor2_plane is None and pixel_col > 0:
        plane = chozen_z[pixel_row, pixel_col - 1]
        if plane >= 0:
            if neighbor1_plane is None:
                neighbor1_plane = plane
            else:
                neighbor2_plane = plane
    if neighbor2_plane is None and pixel_col < max_col - 1:
        plane = chozen_z[pixel_row, pixel_col + 1]
        if plane >= 0:
            if neighbor1_plane is None:
                neighbor1_plane = plane
            else:
                neighbor2_plane = plane

    if neighbor2_plane is None or neighbor1_plane == neighbor2_plane:
        return max(0,neighbor1_plane - 1) + np.argmax(score[max(0, neighbor1_plane-1):min(max_plane,neighbor1_plane+2), pixel_row, pixel_col])
    elif np.abs(neighbor1_plane - neighbor2_plane) == 1:
        plane = min(neighbor1_plane, neighbor2_plane)
        return max(0,plane) + np.argmax(score[max(0, plane):min(max_plane, plane + 2), pixel_row, pixel_col])
    else:
        return (neighbor1_plane + neighbor2_plane) / 2



def movie_surface_projection(files, reference_channel, position_final_movie, initial_positions_number, output_dir,
                             method, bin_size, build_manifold, only_position, zmin, zmax, airyscan):
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
    positions = list(range(initial_positions_number))
    time_points_number = np.zeros((initial_positions_number, len(files)))
    # Projecting
    projection_files = [[] for position in range(initial_positions_number)]
    zmap_files = [[] for position in range(initial_positions_number)]
    for file_num, file in enumerate(files):
        remove_positions = []
        dims = get_image_dimensions(file)
        for position_num, position in enumerate(positions):
            if position_final_movie[position] == file_num + 1:
                remove_positions.append(position)
            if only_position > 0 and position != only_position - 1:
                continue
            projection_path = os.path.join(output_dir, "position%d_movie%d_projection.npy" % (position,file_num))
            zmap_path = os.path.join(output_dir, "position%d_movie%d_zmap.npy" % (position, file_num))
            projection_files[position].append(projection_path)
            zmap_files[position].append(zmap_path)
            print("Projecting position %d, movie %d" % (position + 1, file_num + 1))
            time_points_number[position, file_num] = dims.T
            if os.path.isfile(projection_path) and os.path.isfile(zmap_path):
                continue
            current_projection = np.zeros((dims.T, dims.C, 1, dims.Y, dims.X))
            current_zmap = np.zeros((dims.T, 1, 1, dims.Y, dims.X))
            projector = read_image_in_chunks(file, series=position_num, dt=1,
                                             apply_function=time_point_surface_projection,
                                             output=[current_projection, current_zmap], axes='TCZYX',
                                             reference_channel=reference_channel, z_map=True, method=method,
                                             bin_size=bin_size, atoh_shift=0, build_manifold=build_manifold, min_z=zmin,
                                             max_z=zmax, airyscan=airyscan)
            for time_point_index, time_point in enumerate(projector):
                print("Projecting timepoint %d" % (time_point_index + 1))
            current_projection = current_projection.reshape((dims.T, dims.C, dims.Y, dims.X))
            np.save(projection_path, current_projection)
            np.save(zmap_path, current_zmap)

        for to_delete in remove_positions:
            positions.remove(to_delete)
    #  Updating meta data and saving projections
    for position in range(initial_positions_number):
        if only_position > 0 and position != only_position - 1:
            continue
        former_metadata = get_image_metadata(files[0], series=position)
        new_metadata = update_projection_metadata(former_metadata, np.sum(time_points_number[position, :]),
                                                  series=position)
        movie_projection = concatenate_time_points(projection_files[position])
        save_tiff(os.path.join(output_dir, "position%d.tif" %(position+1)), movie_projection,
                  metadata=new_metadata, axes="TCYX",data_type="uint16")
        movie_zmap = np.concatenate([np.load(zmap_files[position][i]) for i in range(len(zmap_files[position]))], axis=0)
        np.save(os.path.join(output_dir, "zmap_position%d.npy" %(position+1)), movie_zmap)
    # Saving stage location (to correct for stage movement between movies)
    save_stage_positions(files, position_final_movie, initial_positions_number, output_dir)
    # Removing timepoints projection
    for position_files in projection_files + zmap_files:
        for projection_file in position_files:
            os.remove(projection_file)


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
    for position in range(initial_positions_number):
        if position_final_movie[position] == 1:
            positions.remove(position)
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


def large_image_projection(input_dir, output_dir, input_file_name, position=0, reference_channel=0, chunk_size=0,
                           bin_size=1, channels_shift=0, min_z=0, max_z=0, method="", build_manifold=False,
                           airyscan=False):
    path = os.path.join(input_dir, input_file_name)
    if not os.path.exists(path):
        return 0
    dims = get_image_dimensions(path)
    projection = np.zeros((1, dims.C, 1, dims.Y, dims.X))
    zmap = np.zeros((1, 1, 1, dims.Y, dims.X))
    projector = read_image_in_chunks(path, dx=chunk_size, dy=chunk_size,
                                     apply_function=time_point_surface_projection,
                                     output=[projection, zmap], axes='TCZYX', min_z=min_z, max_z=max_z,
                                     reference_channel=reference_channel, series=position, z_map=True, method=method,
                                     bin_size=bin_size, atoh_shift=channels_shift, build_manifold=build_manifold,
                                     airyscan=airyscan)
    for chunk_num, chunk in enumerate(projector):
        print("Projecting chunk %d" % (chunk_num + 1), flush=True)
    projection = projection.reshape((dims.C, dims.Y, dims.X))
    zmap = zmap.reshape((1, dims.Y, dims.X))
    postfix = '.' + input_file_name.split('.')[-1]
    projection_file_name = os.path.join(output_dir, input_file_name.replace(postfix, "_projection.tif"))
    zmap_filename = os.path.join(output_dir, input_file_name.replace(postfix, "_zmap.npy"))
    save_tiff(projection_file_name, projection, axes="CYX", data_type="uint16")
    np.save(zmap_filename, zmap)


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
    parser.add_option("-c", "--chunk-size", dest="chunk_size",
                      help="Chunk size for large fixed sample projection [default: the whole image]",
                      type=int, default=0)
    parser.add_option("--method", dest="method",
                      help="Projection method [default: max_averages]",
                      default="max_averages")
    parser.add_option("--fixed", dest="fixed_sample", action="store_true",
                      help="Projection for a fixed sample [default: Movie projection]",
                      default=False)
    parser.add_option("--file", dest="file_name", help="File name (for fixed sample projection)")
    parser.add_option("-b", "--bin-size", dest="bin_size",
                      help="Bin size for average/std calculation [default: 1]",
                      type=int, default=1)
    parser.add_option("--manifold", dest="build_manifold",
                      help="If true will build a continuous manifold instead of doing a pixel-wise projection [default:False]",
                      default=False, action="store_true")
    parser.add_option("--only-position", dest="only_position",
                      help="Project only the given position [default: all positions]",
                      type=int, default=0)
    parser.add_option("--airyscan", dest="airyscan",
                      help="If true will tread as an airyscan processed image (different intensities) [default:False]",
                      default=False, action="store_true")
    parser.add_option("--min-z", dest="zmin",
                      help="First z surface to consider for projection [default: all surfaces]",
                      type=int, default=0)
    parser.add_option("--max-z", dest="zmax",
                      help="Last z surface to consider for projection [default: all surfaces]",
                      type=int, default=0)
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
    reference_channel = options.reference_channel
    bin_size = options.bin_size
    build_manifold = options.build_manifold
    only_position = options.only_position
    method = options.method
    zmin = options.zmin
    zmax=options.zmax
    airyscan = options.airyscan
    if options.fixed_sample:
        file_name = options.file_name
        chunk_size = options.chunk_size
        large_image_projection(input_dir, output_dir, file_name, position=only_position,
                               reference_channel=reference_channel, chunk_size=chunk_size,
                               bin_size=bin_size, method=method,build_manifold=build_manifold, min_z=zmin, max_z=zmax,
                               airyscan=airyscan)

    else:
        movie_number = options.movie_number
        position_final_movie = options.position_final_movie
        if not options.position_final_movie:
            position_final_movie = [movie_number]*position_number
        else:
            position_final_movie = list(literal_eval(options.position_final_movie))
        files = [os.path.join(input_dir,"m%d.czi" %(i + 1)) for i in range(movie_number)]
        movie_surface_projection(files, reference_channel, position_final_movie, position_number, output_dir,
                                 method, bin_size, build_manifold, only_position, zmin, zmax, airyscan)
    exit(0)



