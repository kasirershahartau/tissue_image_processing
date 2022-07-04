# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 12:53:47 2021

@author: Shahar Kasirer, Anastasia Pergament 

Methods to analyze cells    
"""
import os.path
import shutil
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage.filters import maximum_filter, minimum_filter
from skimage.draw import line, disk
from skimage.measure import regionprops_table, regionprops
from skimage.measure import label as label_image_regions
from skimage.registration import phase_cross_correlation
import zipfile


MIN_CELL_AREA = 100
MAX_CELL_AREA = 6000
HC_TYPE = 1
SC_TYPE = 2
INVALID_TYPE = 0
MAX_SEG_LINE_LENGTH = 100
CELL_INFO_SPECS = {"area": 0,
                   "perimeter": 0,
                   "label": 0,
                   "cx": 0,
                   "cy": 0,
                   "neighbors": set(),
                   "n_neighbors": 0,
                   "valid": 0,
                   "type": "TBA",
                   "bounding_box_min_row": 0,
                   "bounding_box_min_col": 0,
                   "bounding_box_max_row": 0,
                   "bounding_box_max_col": 0,
                   "empty_cell": 0}
EVENTS_INFO_SPEC = {"type": "TBA",
                    "start_frame": 0,
                    "end_frame": 0,
                    "start_pos_x": 0,
                    "start_pos_y": 0,
                    "end_pos_x": 0,
                    "end_pos_y": 0,
                    "daughter_pos_x": 0,
                    "daughter_pos_y": 0,
                    "cell_id": 0,
                    "daughter_id": 0}


TRACK_COLOR = (0, 1, 0)
NEIGHBORS_COLOR = (1, 1, 1)
HC_COLOR = (1, 0, 1)
SC_COLOR = (1, 1, 0)
EVENT_TYPES = ["division", "delamination", "differentiation", "delete event"]
EVENTS_COLOR = {"division": (0,0,1), "delamination": (1,0,0), "differentiation": (0,1,1)}


def make_df(number_of_lines, specs):
    """
    Creates a table based on pandas DataFrame where each line holds the information in "specs".
    """
    dtypes = np.dtype([(name, type(val)) for name, val in specs.items()])
    if number_of_lines > 0:
        arr = np.empty(number_of_lines, dtype=dtypes)
        df = pd.DataFrame.from_records(arr, index=np.arange(number_of_lines))
        for name, val in specs.items():
            df[name] = [set() for i in range(number_of_lines)] if isinstance(val, set) else\
                       [list() for i in range(number_of_lines)] if isinstance(val, list) else val
    else:
        df = pd.DataFrame(columns=specs.keys())
    return df


def get_temp_directory(name):
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
    SPECIAL_FEATURES = ["roundness", "neighbors from the same type", "HC neighbors", "SC neighbors", "contact length",
                        "HC contact length", "SC contact length"]
    SPECIAL_X_ONLY_FEATURES = ["psi6"]
    SPECIAL_Y_ONLY_FEATURES = ["histogram"]
    CELL_TYPES = ["all", "HC", "SC"]

    def __init__(self, number_of_frames, data_path):
        self.number_of_frames = number_of_frames
        self.cells_info = None
        self.labels = None
        self.cell_types = None
        self.events = make_df(0, EVENTS_INFO_SPEC)
        self.drifts = np.zeros((number_of_frames, 2))
        self.cells_info_frame = 0
        self.labels_frame = 0
        self.cell_types_frame = 0
        self.data_path = data_path
        self.working_dir = self.initialize_working_space()
        self.last_added_line = []
        self._neighbors_labels = []
        self._label_before_line_addition = 0
        self.last_action = []
        self._finished_last_line_addition = True
        self._labels_copy_for_line_addition = None
        self.cells_number = 0
        self.valid_frames = np.ones((number_of_frames,)).astype(int)

    def __del__(self):
        shutil.rmtree(self.working_dir)

    def is_valid_frame(self, frame):
        if 0 < frame <= self.number_of_frames:
            return self.valid_frames[frame - 1]
        else:
            return 0

    def set_validity_of_frame(self, frame, valid=True):
        if 0 < frame <= self.number_of_frames:
            self.valid_frames[frame - 1] = int(valid)
        return 0

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
        self.remove_labels()
        self.remove_cell_types()
        self.remove_cells_info()


    def set_labels(self, frame_number, labels, reset_data=False):
        if frame_number != self.labels_frame:
            self.save_labels()
            self.labels_frame = frame_number
        if reset_data:
            self.reset_frame_data()
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

    def get_cells_number(self):
        if self.cells_info is None:
            return self.cells_number
        else:
            return max(self.cells_number, self.cells_info.label.max())

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

    def get_cells_features(self, frame):
        cells_info = self.get_cells_info(frame)
        if cells_info is not None:
            return list(cells_info.columns)
        return []

    def get_events(self):
        return self.events

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

    def get_cell_id_by_position(self, frame, pos):
        labels = self.get_labels(frame)
        cells_info = self.get_cells_info(frame)
        if labels is None or cells_info is None:
            return 0
        x, y = pos
        cell_idx = labels[y, x] - 1
        if cell_idx < 0:
            return 0
        return cells_info.label[cell_idx]

    def add_event(self, event_type, start_frame, end_frame, start_pos, end_pos, second_end_pos=None):
        """
        Adding new event to records.
        @param resulting_cells_id: only for cell division
        """
        if start_frame is None:
            return 0
        if event_type == "delete event":
            self.delete_event(start_frame, start_pos)
            return 0
        start_cell_id = self.get_cell_id_by_position(start_frame, start_pos)
        end_cell_id = self.get_cell_id_by_position(end_frame, end_pos)
        if start_cell_id != end_cell_id and event_type == "differentiation":
            self.fix_cell_label(end_frame, end_pos, start_cell_id)
        new_event = {"type": event_type,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_pos_x": start_pos[0],
                "start_pos_y": start_pos[1],
                "end_pos_x": end_pos[0],
                "end_pos_y": end_pos[1],
                "daughter_pos_x": 0,
                "daughter_pos_y": 0,
                "cell_id": start_cell_id,
                "daughter_id": 0}
        if second_end_pos is not None:
            second_cell_id = self.get_cell_id_by_position(end_frame, second_end_pos)
            if start_cell_id != end_cell_id and start_cell_id == second_cell_id:
                second_cell_id = end_cell_id
            new_event["daughter_pos_x"] = second_end_pos[0]
            new_event["daughter_pos_y"] = second_end_pos[1]
            new_event["daughter_id"] = second_cell_id
        self.events = self.events.append(new_event, ignore_index=True)
        return 0

    def delete_event(self, start_frame, start_pos):
        cell_id = self.get_cell_id_by_position(start_frame, start_pos)
        to_delete = self.events.query("start_frame == %d and (cell_id == %d or daughter_id == %d)" %(start_frame, cell_id, cell_id))
        if to_delete.size > 0:
            self.events.drop(to_delete.index, inplace=True)
        return 0

    def draw_events(self, frame, radius=5):
        labels = self.get_labels(frame)
        if labels is None:
            return 0
        final_image = np.zeros((3,) + labels.shape)
        for event_idx, event in self.events.iterrows():
            color = EVENTS_COLOR[event.type]
            if event.start_frame <= frame <= event.end_frame:
                cell_label = event.cell_id
                cell = self.get_cell_data_by_label(cell_label, frame)
                if cell is None or cell.empty_cell == 1:
                    continue
                else:
                    rr, cc = disk((cell.cy, cell.cx), radius, shape=labels.shape)
                    for i in range(3):
                        final_image[i, rr, cc] = color[i]
                    if event.type == "division":
                        resulting_cell_label = event.daughter_id
                        resulting_cell = self.get_cell_data_by_label(resulting_cell_label, frame)
                        if resulting_cell is not None and resulting_cell.empty_cell == 0:
                            for i in range(3):
                                rr, cc = disk((resulting_cell.cy, resulting_cell.cx), radius, shape=labels.shape)
                                final_image[i, rr, cc] = color[i]
        return final_image

    def calculate_frame_cellinfo(self, frame_number, hc_marker_image=None, hc_threshold=0.01, use_existing_types=False):
        """
        Functions to calculate and organize the cell information.
        """
        labels = self.get_labels(frame_number)
        if labels is None:
            return 0
        number_of_cells = int(np.max(labels))
        if number_of_cells == 0:
            return 0
        cells_info = make_df(number_of_cells, CELL_INFO_SPECS)
        cell_types = self.get_cell_types(frame_number)
        if use_existing_types and cell_types is not None:
            hc_marker_image = (cell_types == HC_TYPE).astype(float)
        if hc_marker_image is not None:
            properties = regionprops_table(labels, intensity_image=hc_marker_image,
                                           properties=['label', 'area', 'perimeter', 'centroid', 'mean_intensity', 'bbox'])
        else:
            properties = regionprops_table(labels, properties=['label', 'area', 'perimeter', 'centroid', 'bbox'])
        cell_indices = properties['label'] - 1
        cells_info.at[cell_indices, "label"] = properties['label']
        cells_info.at[cell_indices, "area"] = properties['area']
        cells_info.at[cell_indices, "perimeter"] = properties['perimeter']
        cells_info.at[cell_indices, "cx"] = properties['centroid-1']
        cells_info.at[cell_indices, "cy"] = properties['centroid-0']
        cells_info.at[cell_indices, "bounding_box_min_row"] = properties['bbox-0']
        cells_info.at[cell_indices, "bounding_box_min_col"] = properties['bbox-1']
        cells_info.at[cell_indices, "bounding_box_max_row"] = properties['bbox-2']
        cells_info.at[cell_indices, "bounding_box_max_col"] = properties['bbox-3']
        areas = cells_info.area.to_numpy()
        cells_info.at[:, "valid"] = np.logical_and(areas < MAX_CELL_AREA, areas > MIN_CELL_AREA)
        self.set_cells_info(frame_number, cells_info)
        self.find_neighbors(frame_number)
        if hc_marker_image is not None:
            self.calc_cell_types(hc_marker_image, frame_number, properties, hc_threshold)

    def get_cell_data_by_label(self, cell_id, frame):
        cells_info = self.get_cells_info(frame)
        if cells_info is None:
            return None
        cells_with_matching_label = (cells_info.label == cell_id).to_numpy()
        if cells_with_matching_label.any():
            return cells_info.iloc[np.argmax(cells_with_matching_label)]
        else:
            return None

    def plot_single_cell_data(self, cell_id, feature, ax):
        current_cell_info_frame = self.cells_info_frame
        t = []
        data = []
        for frame in range(1, self.number_of_frames+1):
            cell = self.get_cell_data_by_label(cell_id, frame)
            if cell is not None:
                t.append((frame-1)*15)
                data.append(cell[feature])
        ax.plot(t, data, '*')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel(feature)
        ax.set_title("%s of cell number %d" % (feature, cell_id))
        self.get_cells_info(current_cell_info_frame)
        return pd.DataFrame({"Time": t, feature: data})

    def get_frame_data(self, frame, feature, valid_cells, special_features, cells_type='all', for_histogram=False):
        if feature in special_features:
            if feature == "psi6":
                second_order_neighbors = self.find_second_order_neighbors(frame, valid_cells, cell_type=cells_type)
                data = self.calc_psin(frame, valid_cells, second_order_neighbors, n=6, for_histogram=for_histogram)
            elif feature == "roundness":
                data = self.calculate_cells_roundness(valid_cells)
            elif feature == "neighbors from the same type":
                data = self.calculate_n_neighbors_from_type(frame, valid_cells, cell_type='same')
            elif feature == "HC neighbors":
                data = self.calculate_n_neighbors_from_type(frame, valid_cells, cell_type='HC')
            elif feature == "SC neighbors":
                data = self.calculate_n_neighbors_from_type(frame, valid_cells, cell_type='SC')
            elif feature == "roundness":
                data = self.calculate_cells_roundness(valid_cells)
            elif "contact length" in feature:
                if 'HC' in feature:
                    neighbors_type = 'HC'
                elif 'SC' in feature:
                    neighbors_type = 'SC'
                else:
                    neighbors_type = 'all'
                if for_histogram:
                    data = np.empty((0,))
                else:
                    data = np.empty((valid_cells.shape[0],))
                    i = 0
                labels = self.get_labels(frame)
                max_filtered_label = maximum_filter(labels, (2,2), mode='constant')
                min_filtered_label = minimum_filter(max_filtered_label, (2,2), mode='constant')

                for index, cell in valid_cells.iterrows():
                    _, contact_lengths = self.calculate_contact_length(frame, cell, max_filtered_label,
                                                                       min_filtered_label,
                                                                       cell_type=neighbors_type)
                    if for_histogram:
                        data = np.hstack((data, contact_lengths))
                    else:
                        data[i] = np.sum(contact_lengths)
                        i += 1

            else:
                return None, "Not implemented yet..."
        else:
            data = valid_cells[feature].to_numpy()
        return data, ""


    def plot_single_frame_data(self, frame, x_feature, y_feature, ax, cells_type='all'):
        cell_info = self.get_cells_info(frame)
        y_data = None
        if cell_info is None:
            return None, "No frame data is available"
        if cells_type == "all":
            valid_cells = cell_info.query("valid == 1")
        else:
            valid_cells = cell_info.query("valid == 1 and type ==\"%s\"" % cells_type)
        plotted = False
        x_data, msg = self.get_frame_data(frame, x_feature, valid_cells,
                                          self.SPECIAL_FEATURES + self.SPECIAL_X_ONLY_FEATURES, cells_type)
        if x_data is None:
            return None, msg
        if y_feature == "histogram":
            ax.hist(x_data)
            ax.set_xlabel(x_feature)
            ax.set_ylabel('frequency')
            title = "%s histogram for frame %d" % (x_feature, frame)
            res = pd.DataFrame({"Frame":[frame]*np.size(x_data), x_feature: x_data})
            plotted = True
        else:
            y_data, msg = self.get_frame_data(frame, y_feature, valid_cells,
                                          self.SPECIAL_FEATURES + self.SPECIAL_Y_ONLY_FEATURES, cells_type)
            if y_data is None:
                return None, msg

        if not plotted:
            ax.plot(x_data, y_data, '*')
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            title = "%s vs %s for frame %d" % (x_feature, y_feature, frame)
            res = pd.DataFrame({"Frame": [frame] * np.size(x_data), x_feature: x_data, y_feature: y_data})
        if cells_type != 'all':
            title += " for %s only" % (cells_type)
        ax.set_title(title)
        return res, ""

    def plot_compare_frames_data(self, frames, feature, ax, cells_type='all'):
        data = []
        err = []
        n_results = []
        current_cell_info_frame = self.cells_info_frame
        for frame in frames:
            cell_info = self.get_cells_info(frame)
            if cell_info is None:
                return None, "No frame data is available for frame %d" % frame
            if cells_type == "all":
                valid_cells = cell_info.query("valid == 1")
            else:
                valid_cells = cell_info.query("valid == 1 and type ==\"%s\"" % cells_type)
            raw_data, msg = self.get_frame_data(frame, feature, valid_cells,
                                              self.SPECIAL_FEATURES + self.SPECIAL_X_ONLY_FEATURES + self.SPECIAL_Y_ONLY_FEATURES,
                                                cells_type, for_histogram=True)
            if raw_data is None:
                return None, msg
            data.append(np.average(raw_data))
            err.append(np.std(raw_data)/np.sqrt(np.size(raw_data)))
            n_results.append(np.size(raw_data))
        x_pos = np.arange(len(frames))
        x_labels = ["frame %d (N = %d)" % (f, n) for f,n in zip(frames, n_results)]
        ax.bar(x_pos, data, yerr=err, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel(feature)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        title = "%s for different frames" % feature
        if cells_type != 'all':
            title += " for %s only" % cells_type
        ax.set_title(title)
        ax.yaxis.grid(True)
        self.get_cells_info(current_cell_info_frame)
        return pd.DataFrame({"Frame": frames, feature + " average": data, feature + " se": err, "N": n_results}), ""


    @staticmethod
    def calculate_cells_roundness(cells):
        return cells.eval("perimeter ** (3/2) / ( area * 3.14 ** (1/2) * 6 )").to_numpy()

    def calculate_n_neighbors_from_type(self, frame, cells, cell_type='same'):
        cell_info = self.get_cells_info(frame)
        if cell_info is None:
            return None
        neighbors_from_type = np.zeros((cells.shape[0],))
        index = 0
        for i, row in cells.iterrows():
            if cell_type == 'same':
                look_for_type = row.type
            else:
                look_for_type = cell_type
            neighbors = np.array(list(row.neighbors))
            neighbors_from_type[index] = np.sum((cell_info.type[neighbors - 1] == look_for_type).to_numpy().astype(int))
            index += 1
        return neighbors_from_type

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
            circle = Circle((int(cell.cx), int(cell.cy)), 8) #plot the circle using centroid coordinates and a radius
            ax.add_patch(circle) #make circles in places of centroid
    
    def find_neighbors(self, frame_number, labels_region=None, only_for_labels=None):#finds all the neighbors in the tissue
        if labels_region is None:
            labels = self.get_labels(frame_number)
        else:
            labels = labels_region
        if labels is None:
            return 0
        cells_info = self.get_cells_info(frame_number)
        # Using max pooling with 3X3 kernel so if cell i that has a neighbor with a smaller label it would have at least
        # one pixel labeled as i in the dilated image
        dilated_image = maximum_filter(labels, (3,3), mode='constant')
        if only_for_labels is None:
            working_indices = np.arange(cells_info.shape[0])
        else:
            working_indices = np.array(only_for_labels) - 1
        for cell_index in working_indices:
            cell_label = cell_index + 1
            neighborhood = labels[dilated_image == cell_label]
            neighborhood[neighborhood == cell_label] = 0
            neighbors_labels = np.unique(neighborhood[neighborhood > 0])
            if neighbors_labels.size > 0:
                neighbors_labels = set(neighbors_labels)
                cells_info.at[cell_index, "neighbors"] = cells_info.neighbors[cell_index].union(neighbors_labels)
                for neighbor_label in list(neighbors_labels):
                   self.cells_info.at[neighbor_label-1, "neighbors"].add(cell_label)
        for cell_index in working_indices:
            self.cells_info.at[cell_index, "n_neighbors"] = len(cells_info.neighbors[cell_index])
        return

    def calculate_contact_length(self, frame, cell_info, max_filtered_labels, min_filtered_labels, cell_type='all'):
        cells_info = self.get_cells_info(frame)
        cell_label = cell_info.label
        region_first_row = max(0, cell_info.bounding_box_min_row - 2)
        region_last_row = cell_info.bounding_box_max_row + 2
        region_first_col = max(0, cell_info.bounding_box_min_col - 2)
        region_last_col = cell_info.bounding_box_max_col + 2
        max_filtered_region = max_filtered_labels[region_first_row:region_last_row, region_first_col:region_last_col]
        min_filtered_region = min_filtered_labels[region_first_row:region_last_row, region_first_col:region_last_col]
        neighbor_labels = np.array(list(cell_info.neighbors.copy()))
        if cell_type != 'all':
            neighbors_from_the_right_type = (cells_info.type[np.array(neighbor_labels) - 1] == cell_type).to_numpy()
            neighbor_labels = neighbor_labels[neighbors_from_the_right_type]
        contact_length = []
        for neighbor_label in neighbor_labels:
            max_label = max(cell_label, neighbor_label)
            min_label = min(cell_label, neighbor_label)
            contact_length.append(np.sum(np.logical_and(max_filtered_region == max_label, min_filtered_region == min_label)).astype(int))
        return neighbor_labels, contact_length

    def track_cells(self, initial_frame=1, final_frame=-1, images=None, image_in_memory=False):
        iter = self.track_cells_iterator(initial_frame, final_frame, images, image_in_memory)
        last_frame = initial_frame
        for frame in iter:
            last_frame = frame
        return last_frame

    def track_cells_iterator(self, initial_frame=1, final_frame=-1, images=None, image_in_memory=False):
        if final_frame == -1:
            final_frame = self.number_of_frames
        cells_info = self.get_cells_info(initial_frame)
        if cells_info is None:
            return 0
        unlabeled_cells = (cells_info.label.to_numpy() == 0)
        last_used_label = cells_info.label.max()
        cells_info.at[unlabeled_cells, "label"] = np.arange(last_used_label + 1,
                                                            last_used_label + np.sum(unlabeled_cells.astype(int)) + 1)
        cx_previous_frame = np.copy(cells_info.cx.to_numpy())
        cy_previous_frame = np.copy(cells_info.cy.to_numpy())
        labels_previous_frame = cells_info.label.to_numpy()
        empty_cells_previous_frame = cells_info.empty_cell.to_numpy()
        previous_frame = initial_frame
        self.cells_number = max(self.cells_number, cells_info.label.max())
        use_existing_drifts = (self.drifts > 0).any()
        for frame in range(initial_frame + 1, final_frame + 1):
            if use_existing_drifts:
                cx_previous_frame -= self.drifts[frame - 1, 1]
                cy_previous_frame -= self.drifts[frame - 1, 0]
            elif images is not None:
                if image_in_memory:
                    previous_img = images[previous_frame - 1, :, 0, :, :]
                    current_img = images[frame - 1, :, 0, :, :]
                else:
                    previous_img = images[previous_frame-1, :, 0, :, :].compute()
                    current_img = images[frame-1, :, 0, :, :].compute()
                shift, error, diffphase = phase_cross_correlation(previous_img, current_img, upsample_factor=100)
                self.drifts[frame-1, :] = shift[-2:]
                cx_previous_frame -= shift[-1]
                cy_previous_frame -= shift[-2]

            cells_info = self.get_cells_info(frame)
            labels = maximum_filter(self.get_labels(frame), (3, 3), mode='constant')
            if cells_info is None or labels is None:
                continue
            cells_info.at[:, "label"] = 0
            indices_in_current_frame = -1 * np.ones(cy_previous_frame.shape)
            y_locations = np.round(cy_previous_frame).astype(int)
            x_locations = np.round(cx_previous_frame).astype(int)
            valid_locations = np.logical_and(np.logical_and(np.logical_and(0 <= y_locations, y_locations < labels.shape[0]),
                              np.logical_and(0 <= x_locations, x_locations < labels.shape[1])), empty_cells_previous_frame == 0)
            indices_in_current_frame[valid_locations] = labels[np.round(cy_previous_frame[valid_locations]).astype(int),
                                              np.round(cx_previous_frame[valid_locations]).astype(int)] - 1
            cells_info.at[indices_in_current_frame[indices_in_current_frame >= 0], "label"] = \
                labels_previous_frame[indices_in_current_frame >= 0]
            unlabeled_cells = (cells_info.label.to_numpy() == 0)
            last_used_label = cells_info.label.max()
            cells_info.at[unlabeled_cells, "label"] = np.arange(last_used_label + 1,
                                                                last_used_label + np.sum(
                                                                    unlabeled_cells.astype(int)) + 1)
            self.cells_number = max(self.cells_number, cells_info.label.max())
            cx_previous_frame = np.copy(cells_info.cx.to_numpy())
            cy_previous_frame = np.copy(cells_info.cy.to_numpy())
            labels_previous_frame = cells_info.label.to_numpy()
            empty_cells_previous_frame = cells_info.empty_cell.to_numpy()
            previous_frame = frame
            yield frame
        yield previous_frame
        self.fix_cell_id_in_events()
        return 0

    def fix_cell_label(self, frame, position, new_label):
        if new_label <= 0:
            return 0
        labels = self.get_labels(frame)
        if labels is None:
            return 0
        x, y = position
        cell_idx = labels[y, x] - 1
        if cell_idx < 0:
            return 0
        cells_info = self.get_cells_info(frame)
        if cells_info is not None:
            current_label = cells_info.label[cell_idx]
            cells_with_same_label_index = cells_info.query("label == %d" % new_label).index
            if len(cells_with_same_label_index) > 0:
                cells_info.at[cells_with_same_label_index[0],"label"] = current_label
            cells_info.at[cell_idx, "label"] = new_label
        return 0

    def fix_cell_id_in_events(self):
        for event_idx in self.events.index:
            start_frame = self.events.start_frame[event_idx]
            end_frame = self.events.end_frame[event_idx]
            start_pos = (self.events.start_pos_x[event_idx], self.events.start_pos_y[event_idx])
            end_pos = (self.events.end_pos_x[event_idx], self.events.end_pos_y[event_idx])
            daughter_pos = (self.events.daughter_pos_x[event_idx], self.events.daughter_pos_y[event_idx])
            cell_id = self.get_cell_id_by_position(start_frame, start_pos)
            cell_end_id = self.get_cell_id_by_position(end_frame, end_pos)

            self.events.at[event_idx, "cell_id"] = cell_id
            if daughter_pos != (0,0):
                daughter_id = self.get_cell_id_by_position(end_frame, daughter_pos)
                if cell_id == daughter_id:
                    daughter_id = cell_end_id
                elif cell_end_id != cell_end_id:
                    self.fix_cell_label(end_frame, end_pos, cell_id)
                self.events.at[event_idx, "daughter_id"] = daughter_id
            else:
                if cell_end_id != cell_id:
                    self.fix_cell_label(end_frame, end_pos, cell_id)
        return 0


    def calc_cell_types(self, hc_marker_image, frame_number, properties=None, hc_threshold=0.1):
        cells_info = self.get_cells_info(frame_number)
        labels = self.get_labels(frame_number)
        self.get_cell_types(frame_number)
        max_brightness = np.max(hc_marker_image[hc_marker_image > 0])
        self.cell_types = np.zeros(labels.shape)
        if properties is None:
            for cell_index in range(len(cells_info)):
                if cells_info.valid[cell_index] == 1:
                    cell_pixels = hc_marker_image[labels == cell_index]
                    average_cell_brightness = np.mean(cell_pixels)
                    if average_cell_brightness > hc_threshold*max_brightness:
                        self.cells_info.at[cell_index, "type"] = "HC"
                        self.cell_types[labels == cell_index] = HC_TYPE
                    else:
                        self.cells_info.at[cell_index, "type"] = "SC"
                        self.cell_types[labels == cell_index] = SC_TYPE
        else:
            cell_indices = properties['label'] - 1
            threshold = hc_threshold * max_brightness
            atoh_intensities = properties["mean_intensity"]
            HC_indices = cell_indices[atoh_intensities > threshold]
            SC_indices = cell_indices[atoh_intensities <= threshold]
            self.cells_info.at[HC_indices, "type"] = "HC"
            self.cells_info.at[SC_indices, "type"] = "SC"
            self.cell_types[np.isin(labels, HC_indices+1)] = HC_TYPE
            self.cell_types[np.isin(labels, SC_indices+1)] = SC_TYPE
        invalid_cells = np.argwhere(self.cells_info.valid.to_numpy() == 0).flatten()
        self.cells_info.at[invalid_cells, "type"] = "invalid"
        self.cell_types[np.isin(labels, invalid_cells+1)] = INVALID_TYPE

    def find_second_order_neighbors(self, frame, cells=None, cell_type='all'):
        cell_info = self.get_cells_info(frame)
        if cell_info is None:
            return None
        if cells is None:
            cells = cell_info.query("valid == 1")
        second_order_neighbors = [set()]*cells.shape[0]
        index = 0
        for i, row in cells.iterrows():
            for neighbor in list(row.neighbors):
                if cell_info.valid[neighbor - 1] == 0:
                    continue
                second_neighbors = np.array(list(cell_info.neighbors[neighbor - 1]))
                if cell_type == 'all':
                    valid_neighbors = second_neighbors[(cell_info.valid[second_neighbors - 1] == 1).to_numpy()]
                else:
                    second_neighbors_info = cell_info.iloc[second_neighbors - 1]
                    valid_neighbors_info = second_neighbors_info.query("valid == 1 and type == \"%s\"" % cell_type)
                    valid_neighbors = valid_neighbors_info.index.to_numpy() + 1
                second_order_neighbors[index] = second_order_neighbors[index].union(set(valid_neighbors))
            second_order_neighbors[index].difference(row.neighbors)
            if i + 1 in second_order_neighbors[index]:
                second_order_neighbors[index].remove(i + 1)
            index += 1
        return second_order_neighbors

    def calc_psin(self, frame, cells, second_order_neighbors, n=6, for_histogram=False):
        cell_info = self.get_cells_info(frame)
        if cell_info is None:
            return None
        psin = np.zeros(cells.shape[0])
        index = 0
        for i, row in cells.iterrows():
            if for_histogram and len(second_order_neighbors[index]) == 0:
                index += 1
                continue
            cell_x = row.cx
            cell_y = row.cy
            second_order_neigbors_indices = np.array(list(second_order_neighbors[index])) - 1
            neighbors_xs = cell_info.cx[second_order_neigbors_indices].to_numpy()
            neighbors_ys = cell_info.cy[second_order_neigbors_indices].to_numpy()
            if np.size(neighbors_xs) > 0:
                psin[index] = np.abs(np.sum(np.exp(-n * 1j * np.arctan2(neighbors_ys-cell_y, neighbors_xs-cell_x))))/np.size(neighbors_xs)
            else:
                psin[index] = 0
            index += 1
        return psin

    def draw_cell_types(self, frame_number):
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
                rr, cc = line(int(cell.cy), int(cell.cx), int(neighbor.cy), int(neighbor.cx))
                img[rr, cc] = 1
        return np.tile(img, (3,1,1))*np.array(NEIGHBORS_COLOR).reshape((3,1,1))

    def draw_cell_tracking(self, frame_number, cell_label, radius=5):
        labels = self.get_labels(frame_number)
        img = np.zeros(labels.shape)
        if labels is None:
            return img
        cell = self.get_cell_data_by_label(cell_label, frame_number)
        if cell is None or cell.empty_cell == 1:
            return img
        else:
            rr, cc = disk((cell.cy, cell.cx), radius, shape=img.shape)
            img[rr, cc] = 1
        return img[np.newaxis, :,:] * np.array(TRACK_COLOR).reshape((3,1,1))

    def add_segmentation_line(self, frame, point1, point2=None, initial=False, final=False, hc_marker_image=None,
                              hc_threshold=0.1, use_existing_types=False):
        points_too_far = False
        labels = self.get_labels(frame)
        if labels is None:
            return 0
        x1, y1 = point1
        if initial:
            self._labels_copy_for_line_addition = np.copy(labels)
        if point2 is not None:
            x2, y2 = point2
            if (x1-x2)**2 + (y1-y2)**2 > MAX_SEG_LINE_LENGTH**2:
                x1, y1 = point2
                point1 = point2
                point2 = None
                points_too_far = True
                final = True
        if point2 is None:
            x2, y2 = self.find_nearest_segmentation_pixel(self._labels_copy_for_line_addition, point1)
        former_label = np.max(labels[max(y1-1,0):y1+1, max(x1-1, 0):x1+1])
        if initial:
            self.last_added_line.append((x1, y1))
            self.last_action.append("add")
            self._finished_last_line_addition = False
            self._label_before_line_addition = [former_label] if former_label > 0 else []
        elif not final and former_label > 0:
            self._label_before_line_addition.append(former_label)
        rr, cc = line(y1, x1, y2, x2)
        self.labels[rr, cc] = 0
        if self.get_cell_types(frame) is not None:
            self.cell_types[rr, cc] = 0
        if final:
            if len(self._label_before_line_addition) > 0:
                label_before_addition = np.bincount(self._label_before_line_addition).argmax()  # majority vote for former label
                self.update_after_adding_segmentation_line(label_before_addition, frame, hc_marker_image,
                                                           hc_threshold, use_existing_types)
            self._finished_last_line_addition = True
        return int(points_too_far)

    def remove_segmentation_line(self, frame, point1, hc_marker_image=None, hc_threshold=0.1, part_of_undo=False,
                                 use_existing_types=False):
        labels = self.get_labels(frame)
        if labels is None:
            return 0
        point = self.find_nearest_segmentation_pixel(labels, point1, distance_limit=20)
        if point[0] is None:
            return 0
        if not part_of_undo:
            self.last_action.append("remove")
        self._neighbors_labels = []
        labels[labels < 0] -= 1

        def remove_neighboring_points_on_line(last_point, initial_point=False):
            x, y = last_point
            labels[y, x] = -1  # Removing the initial point
            neighborhood = labels[max(0,y-1):y+2, max(0,x-1):x+2]
            unique_neighborhood = np.unique(neighborhood[neighborhood > 0])
            neighbors_relative_indices = np.where(neighborhood == 0)
            neighbors_xs = neighbors_relative_indices[1] + x - 1
            neighbors_ys = neighbors_relative_indices[0] + y - 1
            if initial_point or len(neighbors_xs) == 1:
                if len(unique_neighborhood) > 0:
                    for neighbor in unique_neighborhood:
                        if neighbor not in self._neighbors_labels:
                            self._neighbors_labels.append(neighbor)
                if len(self._neighbors_labels) > 2:  # Reached a third neighbor -> reached the edge
                    labels[y, x] = 0
                    return 0
                for neighbor_x, neighbor_y in zip(neighbors_xs, neighbors_ys):
                    remove_neighboring_points_on_line((neighbor_x, neighbor_y))
            elif len(neighbors_xs) > 1:
                for neighbor in unique_neighborhood:
                    if neighbor not in self._neighbors_labels:
                        labels[y, x] = 0
                        return 0
                remove_neighboring_points_on_line((neighbors_xs[0], neighbors_ys[0]))
            return 0

        remove_neighboring_points_on_line(point, True)
        first_neighbor = self._neighbors_labels[0] if len(self._neighbors_labels) > 0 else 0
        second_neighbor = self._neighbors_labels[1] if len(self._neighbors_labels) > 1 else first_neighbor
        self.update_after_segmentation_line_removal(first_neighbor, second_neighbor, frame,
                                                    hc_marker_image, hc_threshold, part_of_undo, use_existing_types)
        return 0

    def change_cell_type(self, frame, pos):
        labels = self.get_labels(frame)
        if labels is None:
            return 0
        x,y = pos
        cell_idx = labels[y, x] - 1
        if cell_idx < 0:
            return 0
        cells_info = self.get_cells_info(frame)
        if cells_info is not None:
            current_type = cells_info.type[cell_idx]
            new_type = "SC" if (current_type == HC_TYPE) else "HC"
            self.cells_info.at[cell_idx, "type"] = new_type
            if current_type == "invalid":
                self.cells_info.at[cell_idx, "valid"] = 1
        cell_types = self.get_cell_types(frame)
        if cell_types is not None:
            current_type = np.max(cell_types[labels == cell_idx + 1])
            new_type = SC_TYPE if (current_type == HC_TYPE) else HC_TYPE
            self.cell_types[labels == cell_idx + 1] = new_type
        return 0

    def make_invalid_cell(self, frame, pos):
        labels = self.get_labels(frame)
        if labels is None:
            return 0
        x, y = pos
        cell_idx = labels[y, x] - 1
        cells_info = self.get_cells_info(frame)
        if cells_info is not None:
            self.cells_info.at[cell_idx, "type"] = "invalid"
            self.cells_info.at[cell_idx, "valid"] = 0
        cell_types = self.get_cell_types(frame)
        if cell_types is not None:
            self.cell_types[labels == cell_idx + 1] = INVALID_TYPE
        return 0

    def update_after_segmentation_line_removal(self, cell1_label, cell2_label, frame, hc_marker_image=None,
                                               hc_threshold=0.1, part_of_undo=False, use_existing_types=False):
        labels = self.get_labels(frame)
        cell_info = self.get_cells_info(frame)
        cell_types = self.get_cell_types(frame)
        if cell1_label != cell2_label and labels is not None:
            new_label = min(cell1_label, cell2_label)
            labels[labels == cell1_label] = new_label
            labels[labels == cell2_label] = new_label
            removed_line_length = np.sum((labels == -1).astype(int))
            if part_of_undo:
                labels[labels == -1] = new_label
                labels[labels < 0] += 1
                if not self._finished_last_line_addition:
                    self._finished_last_line_addition = True
                    return 0
            if cell_info is not None:
                if new_label > 0:
                    cell1_info = cell_info.iloc[cell1_label - 1]
                    cell2_info = cell_info.iloc[cell2_label - 1]
                    area1 = cell1_info.area
                    area2 = cell2_info.area
                    cell_info.at[new_label - 1, "area"] = area1 + area2
                    cell_info.at[new_label - 1, "perimeter"] = cell1_info.perimeter + cell2_info.perimeter - removed_line_length
                    cell_info.at[new_label - 1, "cx"] = (cell1_info.cx*area1 + cell2_info.cx*area2)/\
                                                        (area1 + area2)
                    cell_info.at[new_label - 1, "cy"] = (cell1_info.cy*area1 + cell2_info.cy*area2)/\
                                                        (area1 + area2)
                    cell_info.at[new_label - 1, "bounding_box_min_row"] = min(cell1_info.bounding_box_min_row,
                                                                              cell2_info.bounding_box_min_row)
                    cell_info.at[new_label - 1, "bounding_box_min_col"] = min(cell1_info.bounding_box_min_col,
                                                                              cell2_info.bounding_box_min_col)
                    cell_info.at[new_label - 1, "bounding_box_max_row"] = max(cell1_info.bounding_box_max_row,
                                                                              cell2_info.bounding_box_max_row)
                    cell_info.at[new_label - 1, "bounding_box_max_col"] = max(cell1_info.bounding_box_max_col,
                                                                              cell2_info.bounding_box_max_col)
                    cell_info.at[new_label - 1, "valid"] = MIN_CELL_AREA < area1 + area2 < MAX_CELL_AREA
                    if use_existing_types and cell_types is not None:
                        hc_marker_image = (cell_types == HC_TYPE).astype(float)
                    if hc_marker_image is not None and not use_existing_types:
                        mean_intensity = np.mean(hc_marker_image[labels == new_label])
                        if mean_intensity > hc_threshold * np.max(hc_marker_image[hc_marker_image>0]):
                            cell_info.at[new_label - 1, "type"] = "HC"
                            if cell_types is not None:
                                cell_types[labels == new_label] = HC_TYPE
                        else:
                            cell_info.at[new_label - 1, "type"] = "SC"
                            if cell_types is not None:
                                cell_types[labels == new_label] = SC_TYPE
                    delete_label = max(cell1_label, cell2_label)
                    new_cell_neighbors = cell_info.neighbors[new_label - 1].union(cell_info.neighbors[delete_label - 1])
                    for neighbor_label in list(new_cell_neighbors.copy()):
                        if delete_label in cell_info.neighbors[neighbor_label - 1]:
                            cell_info.neighbors[neighbor_label - 1].remove(delete_label)
                        cell_info.neighbors[neighbor_label - 1].add(new_label)
                        cell_info.neighbors[new_label - 1].add(neighbor_label)
                        cell_info.at[neighbor_label - 1, "n_neighbors"] = len(cell_info.neighbors[neighbor_label - 1])
                    if delete_label in cell_info.neighbors[new_label - 1]:
                        cell_info.neighbors[new_label - 1].remove(delete_label)
                    cell_info.at[new_label - 1, "n_neighbors"] = len(cell_info.neighbors[new_label - 1])
                    cell_info.at[delete_label - 1, "valid"] = 0
                    cell_info.at[delete_label - 1, "empty_cell"] = 1
                    cell_info.at[delete_label - 1, "neighbors"] = set()
                    cell_info.at[delete_label - 1, "n_neighbors"] = 0
                    cell_info.at[delete_label - 1, "label"] = 0
            else:
                new_label = cell1_label
                if part_of_undo:
                    labels[labels == -1] = new_label
                    labels[labels < 0] += 1
                    if not self._finished_last_line_addition:
                        self._finished_last_line_addition = True
                        return 0
        return 0

    def update_after_adding_segmentation_line(self, cell_label, frame, hc_marker_image=None, hc_threshold=0.1,
                                              use_existing_types=False):
        labels = self.get_labels(frame)
        cell_info = self.get_cells_info(frame)
        cell_types = self.get_cell_types(frame)
        if labels is not None:
            if cell_info is None:
                new_label = np.max(labels) + 1
                cell_indices = np.argwhere(labels == cell_label)
                bounding_box_min_row = np.min(cell_indices[:, 0])
                bounding_box_min_col = np.min(cell_indices[:, 1])
                bounding_box_max_row = np.max(cell_indices[:, 0]) + 1
                bounding_box_max_col = np.max(cell_indices[:, 1]) + 1
            else:
                empty_indices = np.argwhere(cell_info.empty_cell.to_numpy() == 1)
                if len(empty_indices) > 0:
                    new_label = empty_indices[0,0] + 1
                else:
                    new_label = cell_info.shape[0] + 1
                cell = cell_info.iloc[cell_label - 1]
                bounding_box_min_row = cell.bounding_box_min_row
                bounding_box_min_col = cell.bounding_box_min_col
                bounding_box_max_row = cell.bounding_box_max_row
                bounding_box_max_col = cell.bounding_box_max_col
            region_first_row = max(0,bounding_box_min_row-2)
            region_first_col = max(0,bounding_box_min_col-2)
            region_last_row = bounding_box_max_row+2
            region_last_col = bounding_box_max_col+2
            cell_region = labels[region_first_row:region_last_row, region_first_col:region_last_col]
            new_region_labels = label_image_regions((cell_region > 0).astype(int), connectivity=1, background=0)
            cell1_label = np.min(new_region_labels[cell_region == cell_label])
            cell2_label = np.max(new_region_labels[cell_region == cell_label])
            cell_region[new_region_labels == cell1_label] = cell_label
            cell_region[new_region_labels == cell2_label] = new_label
            labels[region_first_row:region_last_row, region_first_col:region_last_col] = cell_region
            if cell_info is not None:
                if use_existing_types and cell_types is not None:
                    hc_marker_image = (cell_types == HC_TYPE).astype(float)
                if hc_marker_image is None:
                    properties = regionprops(cell_region)
                else:
                    hc_marker_region = hc_marker_image[region_first_row:region_last_row,
                                                       region_first_col:region_last_col]
                    properties = regionprops(cell_region, intensity_image=hc_marker_region)
                    max_intensity = np.max(hc_marker_image[hc_marker_image > 0])
                for region in properties:
                    if region.label == cell_label:
                        cell_info.at[cell_label - 1, "area"] = region.area
                        cell_info.at[cell_label - 1, "perimeter"] = region.perimeter
                        cell_info.at[cell_label - 1, "cx"] = region.centroid[1] + region_first_col
                        cell_info.at[cell_label - 1, "cy"] = region.centroid[0] + region_first_row
                        cell_info.at[cell_label - 1, "bounding_box_min_row"] = region.bbox[0] + region_first_row
                        cell_info.at[cell_label - 1, "bounding_box_min_col"] = region.bbox[1] + region_first_col
                        cell_info.at[cell_label - 1, "bounding_box_max_row"] = region.bbox[2] + region_first_row
                        cell_info.at[cell_label - 1, "bounding_box_max_col"] = region.bbox[3] + region_first_col
                        cell_info.at[cell_label - 1, "valid"] = MIN_CELL_AREA < region.area < MAX_CELL_AREA
                        if hc_marker_image is not None:
                            mean_intensity = region.mean_intensity
                            if mean_intensity > hc_threshold * max_intensity:
                                cell_info.at[cell_label - 1, "type"] = "HC"
                                if cell_types is not None:
                                    cell_types[labels == cell_label] = HC_TYPE
                            else:
                                cell_info.at[cell_label - 1, "type"] = "SC"
                                if cell_types is not None:
                                    cell_types[labels == cell_label] = SC_TYPE
                    elif region.label == new_label:
                        new_cell_info = {"area": region.area,
                                         "label": new_label,
                                         "perimeter": region.perimeter,
                                         "cx": region.centroid[1] + region_first_col,
                                         "cy": region.centroid[0] + region_first_row,
                                         "bounding_box_min_row": region.bbox[0] + region_first_row,
                                         "bounding_box_min_col": region.bbox[1] + region_first_col,
                                         "bounding_box_max_row": region.bbox[2] + region_first_row,
                                         "bounding_box_max_col": region.bbox[3] + region_first_col,
                                         "valid": MIN_CELL_AREA < region.area < MAX_CELL_AREA,
                                         "empty_cell": 0,
                                         "neighbors": set(),
                                         "n_neighbors": 0}
                        if hc_marker_image is not None:
                            mean_intensity = region.mean_intensity
                            if mean_intensity > hc_threshold * max_intensity:
                                new_cell_info["type"] = "HC"
                                if cell_types is not None:
                                    cell_types[labels == new_label] = HC_TYPE
                            else:
                                new_cell_info["type"] = "SC"
                                if cell_types is not None:
                                    cell_types[labels == new_label] = SC_TYPE
                        cell_info.loc[new_label - 1] = pd.Series(new_cell_info)
                old_cell_neighbors = list(cell_info.neighbors[cell_label - 1].copy())
                for neighbor_label in old_cell_neighbors:
                    cell_info.at[neighbor_label - 1, "neighbors"].remove(cell_label)
                cell_info.at[cell_label - 1, "neighbors"] = set()
                need_to_update_neighbors = list(cell_info.neighbors[cell_label]) + [cell_label, new_label]
                self.find_neighbors(frame, labels_region=cell_region, only_for_labels=need_to_update_neighbors)

    def update_labels(self, frame):
        labels = self.get_labels(frame)
        dilated_image = maximum_filter(labels, (3, 3), mode='constant')
        labels[labels < 0] = dilated_image[labels < 0]
        self.last_action = []
        self._neighbors_labels = (0,0)
        self.last_added_line = []
        return 0

    def undo_last_action(self, frame):
        if len(self.last_action) > 0:
            last = self.last_action.pop(-1)
            if last == "add":
                self.undo_line_addition(frame)
            elif last == "remove":
                self.undo_line_removal(frame)
            return 1
        return 0

    def undo_line_removal(self, frame, hc_marker_image=None):
        labels = self.get_labels(frame)
        if labels is None:
            return
        line_pixels = np.argwhere(labels == -1)
        first_pixel = line_pixels[0]
        neighborhood = labels[first_pixel[0]-1:first_pixel[0]+2, first_pixel[1]-1:first_pixel[1]+2]
        labels[labels < 0] += 1
        self.update_after_adding_segmentation_line(np.max(neighborhood), frame, hc_marker_image)

    def undo_line_addition(self, frame, hc_marker_image=None):
        line_pixel = self.last_added_line.pop(-1)
        self.remove_segmentation_line(frame, line_pixel, hc_marker_image, part_of_undo=True)


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
        if distance_limit > 0:
            return None, None
        else:
            edges = [0, labels.shape[1], 0, labels.shape[2]]
            if nearest_edge < 2:
                return edges[nearest_edge], y
            else:
                return x, edges[nearest_edge]

    def initialize_working_space(self):
        working_dir = get_temp_directory(self.data_path)
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
        self.cell_types_frame = frame_number
        return self.cell_types

    def load_cells_info(self, frame_number):
        file_path = os.path.join(self.working_dir, "frame_%d_data.pkl" % frame_number)
        if os.path.isfile(file_path):
            self.cells_info = pd.read_pickle(file_path)
        else:
            self.cells_info = None
        self.cells_info_frame = frame_number
        return self.cells_info

    def load_events(self):
        file_path = os.path.join(self.working_dir, "events_data.pkl")
        if os.path.isfile(file_path):
            self.events = self.events.append(pd.read_pickle(file_path))
            self.events.drop_duplicates(inplace=True, ignore_index=True)
        return self.events

    def load_drifts(self):
        file_path = os.path.join(self.working_dir, "drifts.npy")
        if os.path.isfile(file_path):
            self.drifts = np.load(file_path)
        return self.drifts

    def load_valid_frames(self):
        file_path = os.path.join(self.working_dir, "valid_frames.npy")
        if os.path.isfile(file_path):
            self.valid_frames = np.load(file_path)
        return self.valid_frames

    def remove_labels(self):
        file_path = os.path.join(self.working_dir, "frame_%d_labels.npy" % self.labels_frame)
        if os.path.isfile(file_path):
            os.remove(file_path)

    def remove_cell_types(self):
        file_path = os.path.join(self.working_dir, "frame_%d_types.npy" % self.cell_types_frame)
        if os.path.isfile(file_path):
            os.remove(file_path)

    def remove_cells_info(self):
        file_path = os.path.join(self.working_dir, "frame_%d_data.pkl" % self.cells_info_frame)
        if os.path.isfile(file_path):
            os.remove(file_path)

    def save_labels(self):
        if self.labels is not None:
            file_path = os.path.join(self.working_dir, "frame_%d_labels.npy" % self.labels_frame)
            if os.path.isfile(file_path):
                os.remove(file_path)
            np.save(file_path, self.labels)
        return 0

    def save_drifts(self):
        if self.drifts is not None:
            file_path = os.path.join(self.working_dir, "drifts.npy")
            if os.path.isfile(file_path):
                os.remove(file_path)
            np.save(file_path, self.drifts)
        return 0

    def save_valid_frames(self):
        if self.drifts is not None:
            file_path = os.path.join(self.working_dir, "valid_frames.npy")
            if os.path.isfile(file_path):
                os.remove(file_path)
            np.save(file_path, self.valid_frames)
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

    def save_events(self):
        file_path = os.path.join(self.working_dir, "events_data.pkl")
        if os.path.isfile(file_path):
            os.remove(file_path)
        self.events.to_pickle(file_path)

    def save(self, path):
        self.save_labels()
        self.save_cells_info()
        self.save_cell_types()
        self.save_events()
        self.save_drifts()
        self.save_valid_frames()
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
        self.load_events()
        self.load_drifts()
        self.load_valid_frames()
        return 0

    def flip_all_data(self):
        for frame in range(1, self.number_of_frames + 1):
            self.flip_frame_data(frame)
        self.drifts[:, [0, 1]] = self.drifts[:, [1, 0]]
        self.events.at[:, ["start_pos_y", "start_pos_x", "end_pos_y", "end_pos_x",
                           "daughter_pos_y", "daughter_pos_x"]] = self.events.loc[:, ["start_pos_x",
                           "start_pos_y", "end_pos_x", "end_pos_y", "daughter_pos_x", "daughter_pos_y"]].values
        self.save_events()
        self.save_drifts()
        return 0

    def flip_frame_data(self, frame):
        """
        A method to flip X and Y data, to fix analyses made on flipped images (mostly to fix a specific bug)
        """
        labels = self.get_labels(frame)
        cell_types = self.get_cell_types(frame)
        cell_info = self.get_cells_info(frame)
        if labels is not None:
            self.labels = np.transpose(labels)
        else:
            print("labels on frame %d are none" % frame)
        if cell_types is not None:
            self.cell_types = np.transpose(cell_types)
        else:
            print("cell types on frame %d are none" % frame)
        # Flipping the required columns in cell info table
        if cell_info is not None:
            self.cells_info.at[:,["cy", "cx", "bounding_box_min_col", "bounding_box_min_row",
                                  "bounding_box_max_col", "bounding_box_max_row"]] = self.cells_info.loc[:,
                                 ["cx", "cy", "bounding_box_min_row", "bounding_box_min_col",
                                  "bounding_box_max_row", "bounding_box_max_col"]].values
        else:
            print("cell info on frame %d are none" % frame)
        self.save_labels()
        self.save_cell_types()
        self.save_cells_info()


