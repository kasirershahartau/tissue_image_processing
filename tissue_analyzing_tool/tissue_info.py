# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 12:53:47 2021

@author: Shahar Kasirer, Anastasia Pergament 

Methods to analyze cells    
"""
import os.path
import shutil
from basic_image_manipulations import watershed_segmentation
import re
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage.filters import maximum_filter
from skimage.draw import line_aa, disk
import zipfile

MIN_CELL_AREA = 100
MAX_CELL_AREA = 6000
HC_THRESHOLD = 50
HC_TYPE = 1
SC_TYPE = 2
CELL_INFO_SPECS = {"area": 0,
                   "label": 0,
                   "cx": 0,
                   "cy": 0,
                   "neighbors": set(),
                   "n_neighbors": 0,
                   "valid": 0,
                   "type": "TBA"}

TRACK_COLOR = (0, 1, 0)
NEIGHBORS_COLOR = (1, 1, 1)
HC_COLOR = (1, 0, 1)
SC_COLOR = (1, 1, 0)

def make_df(number_of_lines, specs):
    """
    Creates a table based on pandas DataFrame where each line holds the information in "specs".
    """
    dtypes = np.dtype([(name, type(val)) for name, val in specs.items()])
    arr = np.empty(number_of_lines, dtype=dtypes)
    df = pd.DataFrame.from_records(arr, index=np.arange(number_of_lines))
    for name, val in specs.items():
        df[name] = [set() for i in range(number_of_lines)] if isinstance(val, set) else val
    return df


def get_temp_direcory(name):
    postfix = 1
    temp_dir = name + "_temp%d" % postfix
    while os.path.exists(temp_dir):
        postfix += 1
        temp_dir = name + "_temp%d" % postfix
    return temp_dir


def pack_archive_with_progress(dirname, zipname):
    # Get total data size in bytes so we can report on progress
    total = 0
    for root, dirs, files in os.walk(dirname):
        for fname in files:
            path = os.path.join(root, fname)
            total += os.path.getsize(path)

    # Get the archive directory name
    basename = os.path.basename(dirname)
    z = zipfile.ZipFile(zipname, 'w', zipfile.ZIP_DEFLATED)
    # Current data byte count
    current = 0
    for root, dirs, files in os.walk(dirname):
        for fname in files:
            path = os.path.join(root, fname)

            percent = 100 * current / total
            yield percent
            z.write(path, arcname=fname)
            current += os.path.getsize(path)
    z.close()


def unpack_archive_with_progress(source, target):
    with zipfile.ZipFile(source, "r") as zip_ref:
        total = len(zip_ref.namelist())
        for index, file in enumerate(zip_ref.namelist()):
            zip_ref.extract(file, target)
            yield 100*index/total


class Tissue(object):
    """
         The tissue class holds the cells of a tissue, and organizes information
         according to cell area and centroid location.
    """
    def __init__(self, number_of_frames, data_path):
        self.number_of_frames = number_of_frames
        self.cells_info = None
        self.labels = None
        self.cell_types = None
        self.cells_info_frame = 0
        self.labels_frame = 0
        self.cell_types_frame = 0
        self.data_path = data_path
        self.working_dir = self.initialize_working_space()

    def __del__(self):
        shutil.rmtree(self.working_dir)

    def reset_all_data(self):
        self.cells_info = None
        self.labels = None
        self.cell_types = None
        self.cells_info_frame = 0
        self.labels_frame = 0
        self.cell_types_frame = 0
        old_working_dir = self.working_dir
        self.working_dir = self.initialize_working_space()
        shutil.rmtree(old_working_dir)
        return 0

    def reset_frame_data(self):
        self.cells_info = None
        self.labels = None
        self.cell_types = None

    def set_labels(self, frame_number, labels):
        if frame_number != self.labels_frame:
            self.save_labels()
            self.labels_frame = frame_number
        self.labels = labels

    def set_cells_info(self, frame_number, cells_info):
        if frame_number != self.cells_info_frame:
            self.save_cells_info()
            self.cells_info_frame = frame_number
        self.cells_info = cells_info

    def set_cell_types(self, frame_number, cell_types):
        if frame_number != self.cell_types_frame:
            self.save_cell_types()
            self.cell_types_frame = frame_number
        self.cell_types = cell_types

    def get_labels(self, frame_number):
        if frame_number != self.labels_frame:
            self.save_labels()
            self.labels_frame = frame_number
            self.labels = self.load_labels(frame_number)
        return self.labels

    def get_cells_info(self, frame_number):
        if frame_number != self.cells_info_frame:
            self.save_cells_info()
            self.cells_info_frame = frame_number
            self.cells_info = self.load_cells_info(frame_number)
        return self.cells_info

    def get_cell_types(self, frame_number):
        if frame_number != self.cell_types_frame:
            self.save_cell_types()
            self.cell_types_frame = frame_number
            self.cell_types = self.load_cell_types(frame_number)
        return self.cell_types

    def get_segmentation(self, frame_number):
        labels = self.get_labels(frame_number)
        if labels is None:
            return None
        else:
            return (labels == 0).astype("int")

    def get_cells_number(self, frame_number):
        cells_info = self.get_cells_info(frame_number)
        if cells_info is None:
            return 0
        else:
            return cells_info.label.max()

    def get_cell_by_pixel(self, x, y, frame_number):
        labels = self.get_labels(frame_number)
        cells_info = self.get_cells_info(frame_number)
        if labels is not None and cells_info is not None:
            index = labels[int(y),int(x)]
            if index > 0:
                return cells_info.iloc[index - 1]
            else:
                return pd.Series([])
        else:
            return None

    def get_cells_features(self):
        for frame in range(self.number_of_frames):
            cells_info = self.get_cells_info(frame)
            if cells_info is not None:
                return list(cells_info.columns)
        return []

    def is_segmented(self, frame_number):
        labels = self.get_labels(frame_number)
        return labels is not None

    def is_analyzed(self, frame_number):
        cells_info = self.get_cells_info(frame_number)
        return cells_info is not None

    def is_any_segmented(self):
        for frame in range(self.number_of_frames):
            if self.is_segmented(frame):
                return True
        return False

    def is_any_analyzed(self):
        for frame in range(self.number_of_frames):
            if self.is_analyzed(frame):
                return True
        return False

    def calculate_frame_cellinfo(self, frame_number, hc_marker_image=None):
        """
        Functions to calculate and organize the cell information.
        """
        labels = self.get_labels(frame_number)
        number_of_cells = int(np.max(labels))
        cells_info = make_df(number_of_cells, CELL_INFO_SPECS)
        for index in range(int(np.max(labels))): #establishes range from 0 to largest cell size
            area = labels[labels == index + 1].size #the area of cells. Add one to eliminate 0s for the area
            if MIN_CELL_AREA < area < MAX_CELL_AREA:
                cells_info.at[index, "valid"] = 1
                locations = np.argwhere(labels == index +1) #Establish locations of the cells. Must be with non-zero numbers/
                centroid = locations.mean(axis=0) #find the mean coordinate of the centroids on each axis
                cells_info.at[index, "area"] = area
                cells_info.at[index, "cy"] = centroid[0]
                cells_info.at[index, "cx"] = centroid[1]
        self.set_cells_info(frame_number, cells_info)
        self.find_neighbors(frame_number)
        if hc_marker_image is not None:
            self.calc_cell_types(hc_marker_image, frame_number)

    def calculate_movie_cell_info(self, hc_marker_movie):
        for frame in range(self.number_of_frames):
            self.calculate_frame_cellinfo(frame, hc_marker_movie[frame].compute())
        tracking_last_frame = self.track_cells()
        return tracking_last_frame

    def get_cell_data_by_label(self, cell_id, frame):
        cells_info = self.get_cells_info(frame)
        if cells_info is None:
            return None
        cells_with_matching_label = (cells_info.label == cell_id).to_numpy()
        if cells_with_matching_label.any():
            return cells_info.iloc[np.argmax(cells_with_matching_label)]
        else:
            return None

    def plot_single_cell_data(self, cell_id, feature):
        t = []
        data = []
        for frame in range(1, self.number_of_frames+1):
            cell = self.get_cell_data_by_label(cell_id, frame)
            if cell is not None:
                t.append((frame-1)*15)
                data.append(cell[feature])
        plt.plot(t, data, '*')
        plt.xlabel('Time (minutes)')
        plt.ylabel(feature)
        plt.title("%s of cell number %d" % (feature, cell_id))
        plt.show()

    def plot_centroids(self, frame_number):
        """
        Function to find and label the centroids of the segmented cells.
        """
        cells_info = self.get_cells_info(frame_number)
        if cells_info is None:
            return
        fig, ax = plt.subplots(1) #makes a subplot of this figure, easier to enclose figures
        plt.imshow(self.get_labels(frame_number))
        # shapex, shapey = self.segmentation.shape
        for cell in cells_info.iterrows(): #loop for each cell
            circle = Circle((cell.cx, cell.cy), 8) #plot the circle using centroid coordinates and a radius
            ax.add_patch(circle) #make circles in places of centroid
    
    def find_neighbors(self, frame_number):#finds all the neighbors in the tissue
        labels = self.get_labels(frame_number)
        if labels is None:
            return 0
        cells_info = self.get_cells_info(frame_number)
        # Using max pooling with 3X3 kernel so if cell i that has a neighbor with a smaller label it would have at least
        # one pixel labeled as i in the dilated image
        dilated_image = maximum_filter(labels, (3,3), mode='constant')
        valid_cells = cells_info.valid.to_numpy() == 1
        valid_cells_indices = np.arange(len(cells_info))[valid_cells]
        for cell_index in valid_cells_indices:
            cell_label = cell_index + 1
            neighberhood = labels[dilated_image == cell_label]
            neighberhood[neighberhood == cell_label] = 0
            neighbors_labels = np.unique(neighberhood[neighberhood > 0])
            if neighbors_labels.size > 0:
                valid_neighbors = cells_info.valid[neighbors_labels-1].to_numpy().astype(np.bool)
                neighbors_labels = set(neighbors_labels[valid_neighbors])
                cells_info.at[cell_index, "neighbors"] = cells_info.neighbors[cell_index].union(neighbors_labels)
                for neighbor_label in list(neighbors_labels):
                   self.cells_info.at[neighbor_label-1, "neighbors"].add(cell_label)
        for cell_index in range(len(cells_info)):
            self.cells_info.at[cell_index, "n_neighbors"] = len(cells_info.neighbors[cell_index])
        return

    def track_cells(self, initial_frame=1, final_frame=-1):
        if final_frame == -1:
            final_frame = self.number_of_frames
        cells_info = self.get_cells_info(initial_frame)
        if cells_info is None:
            return 0
        if (cells_info.label.to_numpy() == 0).all():
            cells_info.label = cells_info.index + 1
        cx_previous_frame = cells_info.cx.to_numpy()
        cy_previous_frame = cells_info.cy.to_numpy()
        labels_previous_frame = cells_info.label.to_numpy()
        previous_frame = 0
        for frame in range(initial_frame + 1, final_frame + 1):
            cx_previous_frame, cy_previous_frame = self.get_registration_correction(cx_previous_frame,
                                                                                    cy_previous_frame,
                                                                                    previous_frame, frame)
            cells_info = self.get_cells_info(frame)
            labels = self.get_labels(frame)
            if cells_info is None or labels is None:
                continue
            cells_info.at[:, "label"] = 0

            indices_in_current_frame = labels[np.round(cy_previous_frame).astype(int),
                                              np.round(cx_previous_frame).astype(int)] - 1
            cells_info.at[indices_in_current_frame[indices_in_current_frame >= 0], "label"] = \
                labels_previous_frame[indices_in_current_frame >= 0]
            cx_previous_frame = cells_info.cx.to_numpy()
            cy_previous_frame = cells_info.cy.to_numpy()
            labels_previous_frame = cells_info.label.to_numpy()
            previous_frame = frame
            yield frame
        yield previous_frame

    def get_registration_correction(self, x, y, origin_frame, destination_frame):
        # TODO: implement
        return x,y

    def calc_cell_types(self, hc_marker_image, frame_number): #differenciate with the cells, based on a frequency level in one of the 3 channels
        cells_info = self.get_cells_info(frame_number)
        labels = self.get_labels(frame_number)
        self.get_cell_types(frame_number)
        self.cell_types = np.zeros(labels.shape)
        max_brightness = np.max(hc_marker_image)
        for cell_index in range(len(cells_info)):
            if cells_info.valid[cell_index] == 1:
                cell_pixels = hc_marker_image[labels == cell_index] #the pixels of each cell within the image, based on boundaries in segmentation
                average_cell_brightness = np.mean(cell_pixels) #average brightness of each cell
                if average_cell_brightness > 0.001*HC_THRESHOLD*max_brightness: #if above, it is hair cell, below supporting cell
                    self.cells_info.at[cell_index, "type"] = "HC"
                    self.cell_types[labels == cell_index] = HC_TYPE
                else:
                    self.cells_info.at[cell_index, "type"] = "SC"
                    self.cell_types[labels == cell_index] = SC_TYPE
                
    def draw_cell_types(self, frame_number): #used to actually color in the cells, based on their types
        cell_types = self.get_cell_types(frame_number)
        hc_image = (cell_types == HC_TYPE) * np.array(HC_COLOR).reshape((3, 1, 1))
        sc_image = (cell_types == SC_TYPE) * np.array(SC_COLOR).reshape((3, 1, 1))
        return hc_image + sc_image

    def draw_neighbors_connections(self, frame_number):
        labels = self.get_labels(frame_number)
        img = np.zeros(labels.shape)
        cells_info = self.get_cells_info(frame_number)
        if labels is None or cells_info is None:
            return img
        for index, cell in cells_info.iterrows():
            for neighbor_label in list(cell.neighbors):
                neighbor = cells_info.iloc[neighbor_label - 1]
                rr, cc, val = line_aa(cell.cy, cell.cx, neighbor.cy, neighbor.cx)
                img[rr, cc] = val
        return np.tile(img, (3,1,1))*np.array(NEIGHBORS_COLOR).reshape((3,1,1))

    def draw_cell_tracking(self, frame_number, cell_label, radius=5):
        labels = self.get_labels(frame_number)
        img = np.zeros(labels.shape)
        if labels is None:
            return img
        cell = self.get_cell_data_by_label(cell_label, frame_number)
        if cell is None:
            return img
        else:
            rr, cc = disk((cell.cy, cell.cx), radius, shape=img.shape)
            img[rr, cc] = 1
        return img[np.newaxis, :,:] * np.array(TRACK_COLOR).reshape((3,1,1))

    def add_segmentation_line(self, frame, point1, point2=None):
        labels = self.get_labels(frame)
        if labels is None:
            return 0
        x1, y1 = point1
        if point2:
            x2, y2 = point2
        else:
            x2, y2 = self.find_nearest_segmentation_pixel(labels, point1)
        rr, cc, _ = line_aa(y1, x1, y2, x2)
        self.labels[rr, cc] = 0
        return 0

    def remove_segmentation_line(self, frame, point1):
        labels = self.get_labels(frame)
        if labels is None:
            return 0
        point = self.find_nearest_segmentation_pixel(labels, point1, distance_limit=10)
        if point is None:
            return 0

        def remove_neighboring_points_on_line(last_point, initial_point=False):
            x, y = last_point
            labels[y, x] = -1  # Removing the initial point
            neighborhood = labels[y-1:y+2, x-1:x+2]
            if len(np.unique(neighborhood[neighborhood > 0])) > 2:  # More than 2 cells in neighborhood -> reached the edge
                labels[y, x] = 0
                return 0
            neighbors_relative_indices = np.where(neighborhood == 0)
            neighbors_xs = neighbors_relative_indices[1] + x - 1
            neighbors_ys = neighbors_relative_indices[0] + y - 1
            if initial_point or len(neighbors_xs) == 1:
                for neighbor_x, neighbor_y in zip(neighbors_xs, neighbors_ys):
                    remove_neighboring_points_on_line((neighbor_x, neighbor_y))
            return 0

        remove_neighboring_points_on_line(point, True)
        return 0

    def update_labels(self):
        new_segmentation = 255*(self.labels == 0).astype(int)
        self.set_labels(self.labels_frame, watershed_segmentation(new_segmentation, 0, 1))
        cell_types = self.get_cell_types(self.labels_frame)
        self.get_cells_info(self.labels_frame)
        self.calculate_frame_cellinfo(self.labels_frame, cell_types)
        self.track_cells(initial_frame=max(1, self.labels_frame-1), final_frame=self.labels_frame)
        return 0

    @staticmethod
    def find_nearest_segmentation_pixel(labels, point, distance_limit=-1):
        x, y = point
        if distance_limit > 0:
            distance_from_edge = distance_limit
        else:
            edges_distances = [x, labels.shape[1]-x, y, labels.shape[0]-y]
            nearest_edge = np.argmin(edges_distances)
            distance_from_edge = edges_distances[nearest_edge]
        for distance in range(distance_from_edge):
            for i in [y-distance, y+distance]:
                for j in range(x-distance, x+distance+1):
                    if labels[i, j] == 0:
                        return j, i
            for j in [x-distance, x+distance]:
                for i in range(y-distance, y+distance+1):
                    if labels[i, j] == 0:
                        return j, i
        if distance_limit:
            return None
        else:
            edges = [0, labels.shape[1], 0, labels.shape[2]]
            if nearest_edge < 2:
                return edges[nearest_edge], y
            else:
                return x, edges[nearest_edge]

    def initialize_working_space(self):
        working_dir = get_temp_direcory(self.data_path)
        os.mkdir(working_dir)
        return working_dir

    def load_labels(self, frame_number):
        file_path = os.path.join(self.working_dir, "frame_%d_labels.npy" % frame_number)
        if os.path.isfile(file_path):
            self.labels = np.load(file_path)
        else:
            self.labels = None
        self.labels_frame = frame_number
        return self.labels

    def load_cell_types(self, frame_number):
        file_path = os.path.join(self.working_dir, "frame_%d_types.npy" % frame_number)
        if os.path.isfile(file_path):
            self.cell_types = np.load(file_path)
        else:
            self.cell_types = None
        self.labels_frame = frame_number
        return self.cell_types

    def load_cells_info(self, frame_number):
        file_path = os.path.join(self.working_dir, "frame_%d_data.pkl" % frame_number)
        if os.path.isfile(file_path):
            self.cells_info = pd.read_pickle(file_path)
        else:
            self.cells_info = None
        self.cells_info_frame = frame_number
        return self.cells_info

    def save_labels(self):
        if self.labels is not None:
            file_path = os.path.join(self.working_dir, "frame_%d_labels.npy" % self.labels_frame)
            if os.path.isfile(file_path):
                os.remove(file_path)
            np.save(file_path, self.labels)
        return 0

    def save_cell_types(self):
        if self.cell_types is not None:
            file_path = os.path.join(self.working_dir, "frame_%d_types.npy" % self.cell_types_frame)
            if os.path.isfile(file_path):
                os.remove(file_path)
            np.save(file_path, self.cell_types)
        return 0

    def save_cells_info(self):
        if self.cells_info is not None:
            file_path = os.path.join(self.working_dir, "frame_%d_data.pkl" % self.cells_info_frame)
            if os.path.isfile(file_path):
                os.remove(file_path)
            self.cells_info.to_pickle(file_path)
        return 0

    def save(self, path):
        for percent_done in pack_archive_with_progress(self.working_dir, path.replace(".seg", "") + ".seg"):
            yield percent_done
        return 0

    def load(self, path):
        old_working_dir = self.working_dir
        self.working_dir = self.initialize_working_space()
        for percent_done in unpack_archive_with_progress(path, self.working_dir):
            yield percent_done
        old_files_list = os.listdir(old_working_dir)
        for file in old_files_list:
            if not os.path.exists(os.path.join(self.working_dir, file)):
                os.rename(os.path.join(old_working_dir, file), os.path.join(self.working_dir, file))
        shutil.rmtree(old_working_dir)
        if self.labels_frame > 0:
            self.load_labels(self.labels_frame)
        if self.cell_types_frame > 0:
            self.load_cell_types(self.cell_types_frame)
        if self.cells_info_frame > 0:
            self.load_cells_info(self.cells_info_frame)
        return 0

    # def load(self, path):
    #     old_working_dir = self.working_dir
    #     self.working_dir = self.initialize_working_space()
    #     shutil.unpack_archive(path, self.working_dir)
    #     old_files_list = os.listdir(old_working_dir)
    #     for file in old_files_list:
    #         if not os.path.exists(os.path.join(self.working_dir, file)):
    #             os.rename(os.path.join(old_working_dir, file), os.path.join(self.working_dir, file))
    #     shutil.rmtree(old_working_dir)
    #     yield 99
    #     return 0


