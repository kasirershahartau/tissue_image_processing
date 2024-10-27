# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 12:53:47 2021

@author: Shahar Kasirer

Methods to analyze cells    
"""
import os.path
from shutil import rmtree
import numpy as np
import pandas as pd
import copy
from matplotlib import pyplot as plt
from matplotlib import cm as colormap
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage import convolve1d
from scipy.interpolate import UnivariateSpline
from scipy.spatial import Voronoi, Delaunay
from scipy.io import savemat
from skimage.draw import line, disk
from skimage.measure import regionprops_table
from skimage.measure import label as label_image_regions
from skimage.filters import threshold_local
from skimage.registration import phase_cross_correlation, optical_flow_tvl1
import zipfile
from basic_image_manipulations import read_tiff, blur_image, save_tiff
from time import sleep
import json
import trackpy
from itertools import chain, combinations, filterfalse
import pickle

MAX_SEG_LINE_LENGTH = 100
CELL_INFO_SPECS = {"area": 0,
                   "perimeter": 0,
                   "label": 0,
                   "cx": 0,
                   "cy": 0,
                   "mean_intensity": [0],
                   "neighbors": set(),
                   "n_neighbors": 0,
                   "valid": 0,
                   "type": 0,
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
                    "daughter_id": 0,
                    "significant_frame": 0,
                    "source": "manual"}


TRACK_COLOR = (0, 1, 0)
NEIGHBORS_COLOR = (1, 1, 1)
POS_COLOR = (1, 0, 1)
NEG_COLOR = (1, 1, 0)
MARKING_COLOR = (0.5, 0.5, 0.5)
INVALID_TYPE_INDEX = 255
INVALID_TYPE_NAME = "invalid"
EVENTS_COLOR = {"ablation": (1,1,0), "division": (0,0,1), "delamination": (1,0,0), "differentiation": (0,1,1),
                "promoted differentiation": (1,1,1)}
TRACKING_COLOR_CYCLE = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1)]
PIXEL_LENGTH = 0.1  # in microns. TODO: get from image metadata

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

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

def find_local_maxima(arr, window_size=7):
    blurred = blur_image(arr, 7)
    maxima = maximum_filter(blurred, size=window_size)
    return np.abs(blurred - maxima) < 1e-6

def is_positive_for_type(type, type_index):
    binary_location = (1 << type_index)
    type = np.array(type).astype(np.uint8)
    binary_location = np.ones_like(type).astype(np.uint8)*binary_location
    res = np.bitwise_and(type, binary_location) == binary_location
    # remove invalid cells (which don't belong to any type)
    if hasattr(res, "__len__"):
        res[type == INVALID_TYPE_INDEX] = False
    elif res == INVALID_TYPE_INDEX:
        return False
    return res


def change_type(current_type, type_index, is_positive):
    binary_location = (1 << type_index)
    res = np.array(current_type).astype(np.uint8)
    # make invalid cells (which don't belong to any type) valid
    if hasattr(res, "__len__"):
        res[current_type == INVALID_TYPE_INDEX] = 0
    elif res == INVALID_TYPE_INDEX:
        res = np.array(0).astype(np.uint8)
    binary_location = binary_location*np.ones_like(current_type).astype(np.uint8)
    res = np.bitwise_and(res, np.bitwise_not(binary_location))
    if is_positive:
        res = np.bitwise_or(res, binary_location)
    return res

class Tissue(object):
    """
         The tissue class holds the cells of a tissue, and organizes information
         according to cell area and centroid location.
    """
    SPECIAL_FEATURES = ["shape index", "roundness", "neighbors from the same type", "HC neighbors", "SC neighbors",
                        "HC second neighbors", "SC second neighbors", "second neighbors",
                        "second neighbors from the same type", "contact length",
                        "HC contact length", "SC contact length", "Mean atoh intensity", "Distance from ablation",
                        ]
    SPATIAL_FEATURES = ["HC density", "SC density", "HC type_fraction", "SC type_fraction"]
    SPECIAL_X_ONLY_FEATURES = ["psi6"]
    SPECIAL_Y_ONLY_FEATURES = ["histogram"]
    GLOBAL_FEATURES = ["density", "type_fraction", "total_area", "number_of_cells",
                       "neighbors correlation", "neighbors correlation average",]
    SPECIAL_EVENT_STATISTICS_FEATURES = ["timing histogram", "spatio-temporal correlation"]
    CELL_TYPES = ["all"]
    FITTING_SHAPES = ["ellipse", "circle arc", "line", "spline"]
    EVENT_TYPES = ["ablation", "division", "delamination", "differentiation", "promoted differentiation"]
    ADDITIONAL_EVENT_MARKING_OPTION = ["delete event"]
    ADDITIONAL_EVENT_STATISTICS_OPTIONS = ["overall reference", "overall reference HC", "overall reference SC"]

    def __init__(self, number_of_frames, data_path, max_cell_area=10, min_cell_area=0.1, load_to_memory=False):
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
        self.max_cell_area = max_cell_area
        self.min_cell_area = min_cell_area
        self.shape_fitting_points = None
        self.shape_fitting_results = [dict() for frame in range(self.number_of_frames)]
        self.shape_fitting_normalization = []
        self.stage_locations = self.load_stage_loactions()
        self.height_maps = self.load_height_map()
        self.data_in_memory = load_to_memory
        self.type_names = []
        if load_to_memory:
            self.cell_info_list = [None]*self.number_of_frames
            self.labels_list = [None]*self.number_of_frames
            self.cells_type_list = [None]*self.number_of_frames

    def is_frame_valid(self, frame):
        return self.valid_frames[frame - 1] == 1

    def clean_up(self):
        rmtree(self.working_dir)

    def is_valid_frame(self, frame):
        if 0 < frame <= self.number_of_frames:
            return self.valid_frames[frame - 1]
        else:
            return 0

    def set_validity_of_frame(self, frame, valid=True):
        if 0 < frame <= self.number_of_frames:
            self.valid_frames[frame - 1] = int(valid)
        return 0

    def get_number_of_valid_frames(self):
        return np.sum(self.valid_frames)

    def reset_all_data(self):
        self.cells_info = None
        self.labels = None
        self.cell_types = None
        self.cells_info_frame = 0
        self.labels_frame = 0
        self.cell_types_frame = 0
        self.type_names = []
        if not self.data_in_memory:
            old_working_dir = self.working_dir
            self.working_dir = self.initialize_working_space()
            rmtree(old_working_dir)
        else:
            self.cell_info_list = [None] * self.number_of_frames
            self.labels_list = [None] * self.number_of_frames
            self.cells_type_list = [None] * self.number_of_frames
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

    def set_valid_cell_area(self, min_area, max_area):
        self.min_cell_area = min_area
        self.max_cell_area = max_area

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

    def get_cells_info(self, frame_number, type_name=""):
        if frame_number != self.cells_info_frame:
            self.save_cells_info()
            self.cells_info_frame = frame_number
            self.cells_info = self.load_cells_info(frame_number, type_name=type_name)
        return self.cells_info

    def get_cell_types(self, frame_number):
        if frame_number != self.cell_types_frame:
            self.save_cell_types()
            self.cell_types_frame = frame_number
            self.cell_types = self.load_cell_types(frame_number)
        return self.cell_types

    def type_pos_neg_list_to_indices(self, pos_neg_list):
        types_list = eval(pos_neg_list)
        pos_types = []
        neg_types = []
        for x in types_list:
            if "pos" in x:
                pos_types.append(self.type_name_to_index(x.replace('-pos', '')))
            elif "neg" in x:
                neg_types.append(self.type_name_to_index(x.replace('-neg', '')))
        return pos_types, neg_types

    def type_name_to_index(self, type_name):
        if type_name in self.type_names:
            return self.type_names.index(type_name)
        else:
            return -1

    def type_index_to_name(self, type_index):
        if len(self.type_names) > type_index:
            return self.type_names[type_index]
        else:
            return ""

    def get_cell_type_names(self):
        pos_neg_list = ["%s-pos" % t for t in self.type_names] + ["%s-neg" % t for t in self.type_names]
        full_powerset = powerset(pos_neg_list)

        # filter nonsense combinations
        def filter_nonsense(x):
            for t in self.type_names:
                if ("%s-pos" % t) in x and ("%s-neg" % t) in x:
                    return True
                elif not x:
                    return True
            return False
        return [str(x) for x in filterfalse(filter_nonsense, full_powerset)] + self.CELL_TYPES

    def merge_invalid_neighboring_cells(self, frame):
        labels = self.get_labels(frame)
        cell_types = self.get_cell_types(frame)
        # Find pixels that are inside invalid cell or on the border between invalid cells
        candidate_pixels = maximum_filter(cell_types, (3, 3), mode='constant') == 0
        # Filter only border pixels
        to_remove = np.logical_and(labels == 0, candidate_pixels)
        remove_list = np.argwhere(to_remove)
        while len(remove_list) > 0:
            self.remove_segmentation_line(frame, (remove_list[0,1], remove_list[0,0]))
            labels = self.get_labels(frame)
            to_remove = np.logical_and(labels == 0, candidate_pixels)
            remove_list = np.argwhere(to_remove)
        self.update_labels(frame)
        return 0

    def get_segmentation(self, frame_number):
        labels = self.get_labels(frame_number)
        if labels is None:
            return None
        else:
            return (labels == 0).astype("int")

    def get_cells_number(self):
        if self.cells_info is not None:
            self.cells_number = max(self.cells_number, self.cells_info.label.max())
        return self.cells_number

    def get_cell_by_pixel(self, x, y, frame_number):
        labels = self.get_labels(frame_number)
        cells_info = self.get_cells_info(frame_number)
        if labels is not None and cells_info is not None:
            index = labels[int(y),int(x)]
            if index > 0:
                try:
                    return cells_info.iloc[index - 1]
                except IndexError:
                    return pd.Series([], dtype='float64')
            else:
                return pd.Series([],dtype='float64')
        else:
            return None

    def get_cells_features(self, frame):
        cells_info = self.get_cells_info(frame)
        if cells_info is not None:
            return list(cells_info.columns) + self.SPECIAL_FEATURES
        return []

    def get_events(self):
        return self.events

    def is_segmented(self, frame_number):
        labels = self.get_labels(frame_number)
        return labels is not None

    def is_analyzed(self, frame_number, type_name=""):
        cells_info = self.get_cells_info(frame_number, type_name=type_name)
        return cells_info is not None

    def is_any_segmented(self):
        for frame in range(1, self.number_of_frames + 1):
            if self.is_segmented(frame):
                return True
        return False

    def is_any_analyzed(self, type_name=""):
        for frame in range(1, self.number_of_frames + 1):
            if self.is_analyzed(frame, type_name=type_name):
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
        try:
            cell_id = cells_info.label[cell_idx]
            return cell_id
        except IndexError:
            return 0


    def get_cell_centroid_by_id(self, frame, id):
        cells_info = self.get_cells_info(frame)
        cell = cells_info.query("label == %d and valid == 1 and empty_cell == 0" % id)
        if cell.shape[0] < 1:
            return None
        if cell.shape[0] > 1:
            print("Warning: more than one cell with the same id. frame: %d, id: %d" % (frame, id))
        return cell.cx.values[0], cell.cy.values[0]

    def add_event(self, event_type, start_frame, end_frame, start_pos=None, end_pos=None, second_end_pos=None,
                  start_cell_id=None, daughter_cell_id=None, source='manual'):
        """
        Adding new event to records.
        @param resulting_cells_id: only for cell division
        """
        if start_frame is None:
            return 0
        if event_type == "delete event":
            self.delete_event(start_frame, start_pos)
            return 0
        if start_pos is not None:
            start_cell_id = self.get_cell_id_by_position(start_frame, start_pos)
        else:
            start_pos = self.get_cell_centroid_by_id(start_frame, start_cell_id)
            if start_pos is None:
                return 0
        if end_pos is not None:
            end_cell_id = self.get_cell_id_by_position(end_frame, end_pos)
        else:
            end_cell_id = start_cell_id
            go_back_frames = 0
            while end_pos is None:
                end_pos = self.get_cell_centroid_by_id(end_frame - go_back_frames, start_cell_id)
                go_back_frames += 1
                if end_frame - go_back_frames < start_frame:
                    return 0
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
                "daughter_id": 0,
                "source": source}
        if second_end_pos is not None or daughter_cell_id is not None:
            if daughter_cell_id is None:
                second_cell_id = self.get_cell_id_by_position(end_frame, second_end_pos)
            else:
                second_cell_id = daughter_cell_id
            if second_end_pos is None:
                second_end_pos = self.get_cell_centroid_by_id(end_frame, daughter_cell_id)
                if second_cell_id is None:
                    return 0
            if start_cell_id != end_cell_id and start_cell_id == second_cell_id:
                second_cell_id = end_cell_id
            new_event["daughter_pos_x"] = second_end_pos[0]
            new_event["daughter_pos_y"] = second_end_pos[1]
            new_event["daughter_id"] = second_cell_id
        new_event["significant_frame"] = int(self.find_event_frame(new_event))
        self.events = pd.concat([self.events, pd.DataFrame(new_event, index=[0])], ignore_index=True)
        return 0

    def delete_event(self, start_frame, start_pos):
        cell_id = self.get_cell_id_by_position(start_frame, start_pos)
        to_delete = self.events.query("start_frame == %d and (cell_id == %d or daughter_id == %d)" %(start_frame, cell_id, cell_id))
        if to_delete.size > 0:
            self.events.drop(to_delete.index, inplace=True)
        return 0

    def delete_all_events_in_frame(self, frame, source='all'):
        if source == 'all':
            to_delete = self.events.query("start_frame <= %d <= end_frame" % frame)
        else:
            to_delete = self.events.query("(start_frame <= %d <= end_frame) and source == \"%s\"" % (frame, source))
        if to_delete.size > 0:
            self.events.drop(to_delete.index, inplace=True)
        return 0

    def delete_all_events(self, source='all'):
        if source == 'all':
            self.events = make_df(0, EVENTS_INFO_SPEC)
        else:
            to_delete = self.events.query("source == \"%s\"" % source)
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
                if cell is None or cell.empty_cell.values[0] == 1:
                    continue
                else:
                    rr, cc = disk((cell.cy.values[0], cell.cx.values[0]), radius, shape=labels.shape)
                    for i in range(3):
                        final_image[i, rr, cc] = color[i]
                    if event.type == "division":
                        resulting_cell_label = event.daughter_id
                        resulting_cell = self.get_cell_data_by_label(resulting_cell_label, frame)
                        if resulting_cell is not None and resulting_cell.empty_cell.values[0] == 0:
                            for i in range(3):
                                rr, cc = disk((resulting_cell.cy.values[0], resulting_cell.cx.values[0]), radius, shape=labels.shape)
                                final_image[i, rr, cc] = color[i]
        return final_image

    @staticmethod
    def detect_edge_cells(labels):
        edge_pixels = np.hstack([labels[0,:], labels[:,0], labels[-1,:], labels[:,-1]])
        return np.unique(edge_pixels[edge_pixels > 0]) - 1

    @staticmethod
    def detect_non_sensory_region_cells(cells_info):
        hair_cells = cells_info.query("type == 1 and empty_cell == 0")
        hull = Delaunay(hair_cells.loc[:,["cx","cy"]].to_numpy())
        indices = cells_info.index.array.to_numpy()
        cells_outside_region = hull.find_simplex(cells_info.loc[:,["cx","cy"]].to_numpy()) < 0
        return indices[cells_outside_region]

    def find_valid_frames(self, initial_frame, final_frame):
        initial_frame = max(1, initial_frame)
        final_frame = min(self.number_of_frames, final_frame)
        relevenat_frames = np.arange(initial_frame, final_frame) - 1
        validity = self.valid_frames[relevenat_frames]
        return relevenat_frames[validity == 1] + 1

    def find_events(self, initial_frame, final_frame):
        iter = self.find_events_iterator(initial_frame, final_frame)
        last_frame = initial_frame
        for frame in iter:
            last_frame = frame
        return last_frame

    def find_events_iterator(self, initial_frame=1, final_frame=-1, differentiation_type_name="",
                             differentiation_type_index=0):
        if differentiation_type_name:
            index = self.type_name_to_index(differentiation_type_name)
            if index >= 0:
                differentiation_type_index = index
        if final_frame == -1:
            final_frame = self.number_of_frames
        labels = None
        cells_info = None
        initial_frame -= 1
        while labels is None or cells_info is None:
            initial_frame += 1
            labels = self.get_labels(initial_frame)
            cells_info = self.get_cells_info(initial_frame)
        valid_cells_last_frame = cells_info.query("valid == 1 and empty_cell == 0")
        id_previous_frame = valid_cells_last_frame.label

        HC_id_previous_frame = valid_cells_last_frame.loc[is_positive_for_type(valid_cells_last_frame.type.as_numpy(),
                                                                                 differentiation_type_index)].label


        labels_previous_frame = np.copy(labels)
        edge_cells_id_previous_frame = cells_info.label[self.detect_edge_cells(labels)]
        skipped_frames = 0
        for frame in range(initial_frame + 1, final_frame + 1):
            if not self.valid_frames[frame - 1]:
                skipped_frames += 1
                continue
            adjacent_valid_frames = self.find_valid_frames(frame - 5, frame + 5)
            start_frame = np.min(adjacent_valid_frames)
            end_frame = np.max(adjacent_valid_frames)
            labels = self.get_labels(frame)
            cells_info = self.get_cells_info(frame)
            valid_cells_current_frame = cells_info.query("valid == 1 and empty_cell == 0")
            id_current_frame = valid_cells_current_frame.label
            edge_cells_id_current_frame = cells_info.label[self.detect_edge_cells(labels)]
            HC_id_current_frame = valid_cells_current_frame.loc[is_positive_for_type(valid_cells_current_frame.type.as_numpy(),
                                                                                 differentiation_type_index)].label
            if skipped_frames < 3:
                # Looking for delaminations
                existed_but_now_not = np.setdiff1d(id_previous_frame.values, id_current_frame.values)
                for id in existed_but_now_not:
                    if id in edge_cells_id_previous_frame.values:
                        continue
                    delamination_detected = True
                    neighbors = valid_cells_last_frame.query("label == %d" % id).neighbors
                    if neighbors.shape[0] < 1:
                        continue
                    if neighbors.shape[0] > 1:
                        print("Warning: more than one cell with the same id. frame: %d, cell id: %d" % (frame, id))
                    neighbors = neighbors.values[0]
                    for neighbor in list(neighbors):
                        if neighbor in id_previous_frame:
                            neighbor_id = id_previous_frame[neighbor]
                            if neighbor_id in existed_but_now_not or neighbor_id in edge_cells_id_previous_frame.values:
                                delamination_detected = False
                                break
                        else:
                            delamination_detected = False
                            break
                    if delamination_detected:
                        self.add_event("delamination", start_frame, frame, start_cell_id=id, source='automatic')

                exist_in_both_frames = np.intersect1d(id_current_frame.values, id_previous_frame.values)
                # Looking for differentiations
                differentiating_cells = np.intersect1d(np.setdiff1d(HC_id_current_frame.values, HC_id_previous_frame.values),
                                                       exist_in_both_frames)

                for id in differentiating_cells:
                    differentiation_detected = True
                    neighbors = valid_cells_last_frame.query("label == %d" % id).neighbors
                    if neighbors.shape[0] < 1:
                        continue
                    if neighbors.shape[0] > 1:
                        print("Warning: more than one cell with the same id. frame: %d, cell id: %d" % (frame, id))
                    neighbors = neighbors.values[0]
                    for neighbor in list(neighbors):
                        if neighbor in id_previous_frame:
                            neighbor_id = id_previous_frame[neighbor]
                            if neighbor_id in existed_but_now_not or neighbor_id in edge_cells_id_previous_frame.values:
                                differentiation_detected = False
                                break
                        else:
                            differentiation_detected = False
                            break
                    if differentiation_detected:
                        self.add_event("differentiation", start_frame, end_frame, start_cell_id=id, source='automatic')

                # Looking for divisions
                exist_but_not_before = np.setdiff1d(id_current_frame.values, id_previous_frame.values)
                for id in exist_but_not_before:
                    if id in edge_cells_id_current_frame.values:
                        continue
                    division_detected = False
                    dividing_cell_id = None
                    dividing_cell_position = None
                    cell_centroid = self.get_cell_centroid_by_id(frame, id)
                    centroid_x = int(np.round(cell_centroid[0]))
                    centroid_y = int(np.round(cell_centroid[1]))
                    if self.drifts is not None:
                        drift = self.drifts[frame - 1,:]
                        if drift[0] != np.nan:
                            centroid_x += int(drift[1])
                            centroid_y += int(drift[0])
                    if centroid_x < 0 or centroid_x >= labels_previous_frame.shape[1] or\
                        centroid_y < 0 or centroid_y >= labels_previous_frame.shape[0]:
                        continue
                    label_previous_frame = labels_previous_frame[centroid_y, centroid_x]
                    neighbors = valid_cells_current_frame.query("label == %d" % id).neighbors
                    if neighbors.shape[0] < 1:
                        continue
                    if neighbors.shape[0] > 1:
                        print("Warning: more than one cell with the same id. frame: %d, cell id: %d" % (frame, id))
                    neighbors = neighbors.values[0]
                    for neighbor in list(neighbors):
                        if neighbor in id_current_frame:
                            neighbor_id = id_current_frame[neighbor]
                            if neighbor_id in exist_in_both_frames and neighbor_id not in edge_cells_id_current_frame.values:
                                neighbor_centroid = self.get_cell_centroid_by_id(frame, neighbor_id)
                                neighbor_centroid_x = int(np.round(neighbor_centroid[0]))
                                neighbor_centroid_y = int(np.round(neighbor_centroid[1]))
                                if self.drifts is not None:
                                    drift = self.drifts[frame - 1, :]
                                    if drift[0] != np.nan:
                                        neighbor_centroid_x += int(drift[1])
                                        neighbor_centroid_y += int(drift[0])
                                if neighbor_centroid_x < 0 or neighbor_centroid_x >= labels_previous_frame.shape[1] or \
                                        neighbor_centroid_y < 0 or neighbor_centroid_y >= labels_previous_frame.shape[0]:
                                    continue
                                elif labels_previous_frame[neighbor_centroid_y, neighbor_centroid_x] == label_previous_frame:
                                    division_end_frame = end_frame+1
                                    daughter_end_pos = None
                                    while daughter_end_pos is None:
                                        division_end_frame -= 1
                                        if self.valid_frames[division_end_frame - 1] == 1:
                                            daughter_end_pos = self.get_cell_centroid_by_id(division_end_frame, id)
                                    division_detected = True
                                    dividing_cell_id = neighbor_id
                                    dividing_cell_position = daughter_end_pos
                        else:
                            division_detected = False
                            break
                    if division_detected:
                        self.add_event("division", start_frame, division_end_frame,
                                       start_cell_id=dividing_cell_id, daughter_cell_id=id,
                                       second_end_pos=dividing_cell_position, source='automatic')
            valid_cells_last_frame = valid_cells_current_frame
            id_previous_frame = id_current_frame
            HC_id_previous_frame = HC_id_current_frame
            labels_previous_frame = np.copy(labels)
            skipped_frames = 0
            yield frame
        return 0

    def calc_overall_drift(self):
        overall_drift = np.zeros(self.drifts.shape)
        last_drift_y = 0
        last_drift_x = 0
        for frame in range(self.number_of_frames):
            if self.is_frame_valid(frame + 1):
                last_drift_y += self.drifts[frame, 0]
                last_drift_x += self.drifts[frame, 1]
            overall_drift[frame, 0] = last_drift_y
            overall_drift[frame, 1] = last_drift_x
        return overall_drift

    def calculate_neighbors_correlation_function(self, frame, valid_cells, set_state_by="type", method="neighbors", type_name=""):
        type_index = self.type_name_to_index(type_name)
        if type_index < 0:
            type_index = 0
        if set_state_by == "intensity":
            state = valid_cells.mean_intensity
        elif set_state_by == "type":
            state = pd.Series(data=0, index=valid_cells.index)
            HC_indices = valid_cells.loc[is_positive_for_type(valid_cells.type.to_numpy(), type_index)].index
            state[HC_indices] = 1
        state_avg = np.average(state)
        states_var = np.var(state)
        if method == "neighbors":
            frame_corr = 0
            contacts_counter = 0
            for index, cell in valid_cells.iterrows():
                cell_state = state[index]
                cell_state_minus_average = cell_state - state_avg
                for neighbor_index in list(cell.neighbors):
                    if neighbor_index - 1 in valid_cells.index:
                        neighbor_state = state[neighbor_index - 1]
                        frame_corr += (neighbor_state - state_avg) * cell_state_minus_average
                        contacts_counter += 1
            frame_corr /= (contacts_counter * states_var)

        elif method == "neighbors average":
            neighbors_states = np.zeros((valid_cells.shape[0],))
            i = 0
            for index, cell in valid_cells.iterrows():
                neighbors_state_sum = 0
                valid_neighbors_counter = 0
                for neighbor_index in list(cell.neighbors):
                    if neighbor_index - 1 in valid_cells.index:
                        neighbors_state_sum += state[neighbor_index - 1]
                        valid_neighbors_counter += 1
                if valid_neighbors_counter > 0:
                    neighbors_states[i] = neighbors_state_sum/valid_neighbors_counter
                i += 1
            frame_corr = np.sum((state.to_numpy() - state_avg)*(neighbors_states - np.average(neighbors_states)))/\
                         (valid_cells.shape[0] * np.sqrt(states_var) * np.std(neighbors_states))
        else:
            raise NotImplementedError
        return frame_corr

    def calc_neighborwise_distance(self):
        #skimage.graph.RAG(label_image=None)
        pass

    def calculate_events_correlation_function(self, spatial_bin_size, temporal_bin_size, event_type="all"):
        events = self.get_events()
        if event_type != "all":
            events = events.query("type == \"%s\"" % event_type)
        overall_drift = self.calc_overall_drift()
        frame_shape = self.get_labels(self.labels_frame).shape
        initial_r_bins = frame_shape[1] // spatial_bin_size
        initial_t_bins = self.number_of_frames//temporal_bin_size
        correlation = np.zeros((initial_t_bins, initial_r_bins))
        for index1, event1 in events.iterrows():
            for index2, event2 in events.iterrows():
                if index2 < index1:
                    continue
                x_dist = event1.start_pos_x + overall_drift[event1.start_frame, 1] - event2.start_pos_x - overall_drift[event2.start_frame, 1]
                y_dist = event1.start_pos_y + overall_drift[event1.start_frame, 0] - event2.start_pos_y - overall_drift[event2.start_frame, 0]
                r_dist = np.sqrt(x_dist**2 + y_dist**2)
                t_dist = abs(event1.start_frame - event2.start_frame)
                r_bin = r_dist // spatial_bin_size
                t_bin = t_dist // temporal_bin_size
                if abs(t_bin) >= correlation.shape[0] or abs(r_bin) >= correlation.shape[1]:
                    new_correlation = np.zeros((correlation.shape[0]*2, correlation.shape[1]*2))
                    new_correlation[:correlation.shape[0], :correlation.shape[1]] = correlation
                    correlation = new_correlation
                correlation[int(t_bin), int(r_bin)] = correlation[int(t_bin), int(r_bin)] + 1
        # normalizing by the number of pixels in every distance
        bin_average_dist = spatial_bin_size/2
        for r_bin_index in range(correlation.shape[1]):
            correlation[:,r_bin_index] = correlation[:,r_bin_index]/(2*np.pi*bin_average_dist)
            bin_average_dist += spatial_bin_size
        return correlation/events.shape[0]

    def calculate_frame_cellinfo(self, frame_number):
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
        properties = regionprops_table(labels, properties=['label', 'area', 'perimeter', 'centroid', 'bbox'])
        cell_indices = properties['label'] - 1
        cells_info.loc[cell_indices, "label"] = properties['label']
        cells_info.loc[cell_indices, "area"] = properties['area']
        cells_info.loc[cell_indices, "perimeter"] = properties['perimeter']
        cells_info.loc[cell_indices, "cx"] = properties['centroid-1']
        cells_info.loc[cell_indices, "cy"] = properties['centroid-0']
        cells_info.loc[cell_indices, "bounding_box_min_row"] = properties['bbox-0']
        cells_info.loc[cell_indices, "bounding_box_min_col"] = properties['bbox-1']
        cells_info.loc[cell_indices, "bounding_box_max_row"] = properties['bbox-2']
        cells_info.loc[cell_indices, "bounding_box_max_col"] = properties['bbox-3']
        areas = cells_info.area.to_numpy()
        mean_area = np.mean(areas)
        max_area = self.max_cell_area * mean_area
        min_area = self.min_cell_area * mean_area
        cells_info.loc[:, "valid"] = np.logical_and(areas < max_area, areas > min_area).astype(int)
        self.set_cells_info(frame_number, cells_info)
        self.find_neighbors(frame_number, only_for_labels=cells_info.query("valid == 1").label.to_numpy())
        return 0

    def get_cell_data_by_label(self, cell_id, frame):
        cells_info = self.get_cells_info(frame)
        if cells_info is None:
            return None
        cells_with_matching_label = cells_info.query("label == %d and valid == 1 and empty_cell == 0" % cell_id)
        if cells_with_matching_label.shape[0] > 0:
            return cells_with_matching_label
        else:
            return None

    def plot_single_cell_data(self, cell_id, feature, ax, intensity_images=None, window_radius=0):
        frames = np.arange(1, self.number_of_frames + 1)
        t = (frames - 1)*15
        data, msg = self.get_single_cell_data(cell_id, frames, feature, intensity_images, window_radius)
        t = t[~np.isnan(data)]
        data = data[~np.isnan(data)]
        ax.plot(t, data, '*')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel(feature)
        ax.set_title("%s of cell number %d" % (feature, cell_id))
        return pd.DataFrame({"Time": t, feature: data})

    def get_single_cell_data(self, cell_id, frames, feature, intensity_images=None, window_radius=0):
        current_cell_info_frame = self.cells_info_frame
        data = np.zeros((len(frames),))
        msg = ""
        for index,frame in enumerate(frames):
            cell = self.get_cell_data_by_label(cell_id, frame)
            if self.is_frame_valid(frame) and cell is not None:
                if intensity_images is None:
                    intensity_image = None
                else:
                    intensity_image = intensity_images[index]
                frame_data, msg = self.get_frame_data(frame, feature, cell, special_features=self.SPECIAL_FEATURES,
                                                      spatial_features=self.SPATIAL_FEATURES,
                                                      intensity_img=intensity_image, window_radius=window_radius)
                if msg:
                    return None, msg
                data[index] = frame_data[0]
            else:
                data[index] = np.nan
                msg += "frame %d is invalid\n" % frame
        self.get_cells_info(current_cell_info_frame)
        return data, msg

    def plot_event_related_data(self, cell_id, event_frame, feature, frames_around_event, ax, intensity_images=None):
        event_data = self.events.query("cell_id == %d and start_frame <= %d and end_frame >= %d" %(cell_id, event_frame, event_frame))
        if event_data.shape[0] < 1:
            return None
        else:
            frames = np.arange(max(event_frame - frames_around_event, 0),
                               min(event_frame + frames_around_event + 1, self.number_of_frames + 1))
            t = (frames - 1) * 15
            data, msg = self.get_single_cell_data(cell_id, frames, feature, intensity_images)
            t = t[~np.isnan(data)]
            frames = frames[~np.isnan(data)]
            data = data[~np.isnan(data)]
            ax.plot(t[frames < event_frame], data[frames < event_frame], 'b*', label='before event')
            ax.plot(t[frames >= event_frame], data[frames >= event_frame], 'g*', label='after event')
            title = "%s of cell number %d" % (feature, cell_id)
            event_type = event_data.type.values[0]
            res = {"Time": t, feature: data, "Event type": [event_type]*t.size, "Cell ID": [cell_id]*t.size}
            if event_type == "division":
                daughter_frames = np.arange(event_frame, min(event_frame + frames_around_event + 1, self.number_of_frames + 1))
                t_daughter = (daughter_frames - 1)*15
                daughter_id = event_data.daughter_id.values[0]
                daughter_data, msg = self.get_single_cell_data(daughter_id, daughter_frames, feature)
                t_daughter = t_daughter[~np.isnan(daughter_data)]
                daughter_data = daughter_data[~np.isnan(daughter_data)]
                ax.plot(t_daughter, daughter_data, 'r*', label='daughter cell after event')
                title += " and daughter cell number %d" % daughter_id
                res["Daughter time"] = np.hstack((t_daughter, np.zeros((t.size - t_daughter.size,))))
                res["Daughter data"] = np.hstack((daughter_data,  np.zeros((t.size - daughter_data.size,))))
                res["Daughter ID"] = [daughter_id]*t.size
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel(feature)
            ax.set_title(title)
            ax.legend()
            return pd.DataFrame(res)

    def find_events_frame(self):
        significant_frames = np.zeros((self.events.shape[0],))
        for event_index in range(self.events.shape[0]):
            significant_frames[event_index] = self.find_event_frame(self.events.iloc[event_index])
        self.events["significant_frame"] = significant_frames.astype("int")
        return 0

    def find_event_frame(self, event):
        start_frame = event["start_frame"]
        end_frame = event["end_frame"]
        event_type = event["type"]
        if event_type == "delamination":
            cell_label = event["cell_id"]
            last_valid_frame = start_frame
            for frame in range(start_frame, end_frame+1):
                if self.is_frame_valid(frame):
                    cell = self.get_cell_data_by_label(cell_label, frame)
                    if cell is None or cell.empty_cell.values[0] == 0:
                        return last_valid_frame
                    elif float(cell.area) < self.min_cell_area:
                        return frame
                    last_valid_frame = frame
        if event_type == "division":
            daughter_label = event["daughter_id"]
            last_valid_frame = start_frame
            for frame in range(start_frame, end_frame+1):
                if self.is_frame_valid(frame):
                    cell = self.get_cell_data_by_label(daughter_label, frame)
                    if cell is not None and cell.empty_cell.values[0] == 0:
                        return last_valid_frame
                    last_valid_frame = frame
        if event_type == "differentiation":
            cell_label = event["cell_id"]
            last_valid_frame = start_frame
            for frame in range(start_frame, end_frame+1):
                if self.is_frame_valid(frame):
                    cell = self.get_cell_data_by_label(cell_label, frame)
                    if cell is not None:
                        if cell.type.values[0] == "HC":
                            return last_valid_frame
                    last_valid_frame = frame
        print("Problem with finding event frame")
        return start_frame

    def get_frame_data(self, frame, feature, valid_cells, special_features=[], global_features=[], spatial_features=[],
                      for_histogram=False, reference=None, intensity_img=None, window_radius=0):
        if feature in special_features:
            if feature == "psi6":
                nearest_HCs = self.find_nearest_neighbors_using_voroni_tesselation(valid_cells)
                data = self.calc_psin(frame, valid_cells, nearest_HCs, n=6, for_histogram=for_histogram)
            elif feature == "shape index":
                data = self.calculate_cells_shape_index(valid_cells)
            elif feature == "roundness":
                data = self.calculate_cells_roundness(valid_cells)
            elif feature == "neighbors from the same type":
                data = self.calculate_n_neighbors_from_type(frame, valid_cells, cell_type='same')
            elif feature == "HC neighbors":
                data = self.calculate_n_neighbors_from_type(frame, valid_cells, cell_type='HC', positive_for_type=True)
            elif feature == "SC neighbors":
                data = self.calculate_n_neighbors_from_type(frame, valid_cells, cell_type='HC', positive_for_type=False)
            elif feature == "HC second neighbors":
                data = self.calculate_n_neighbors_from_type(frame, valid_cells, cell_type='HC',positive_for_type=True, second_neighbors=True)
            elif feature == "SC second neighbors":
                data = self.calculate_n_neighbors_from_type(frame, valid_cells, cell_type='HC',positive_for_type=False, second_neighbors=True)
            elif feature == "second neighbors":
                data = self.calculate_n_neighbors_from_type(frame, valid_cells, cell_type='all', second_neighbors=True)
            elif feature == "second neighbors from the same type":
                data = self.calculate_n_neighbors_from_type(frame, valid_cells, cell_type='same', second_neighbors=True)
            elif feature == "Mean atoh intensity":
                data = self.calculate_mean_intensity(frame, valid_cells, intensity_img=intensity_img)
            elif feature == "Distance from ablation":
                data = self.calculate_distance_from_ablation(frame, valid_cells)
            elif "contact length" in feature:
                positive_for_type = True
                if 'HC' in feature:
                    neighbors_type = 'HC'
                elif 'SC' in feature:
                    neighbors_type = 'SC'
                    positive_for_type = False
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
                                                                       cell_type=neighbors_type,
                                                                       positive_for_type=positive_for_type)
                    if for_histogram:
                        data = np.hstack((data, contact_lengths))
                    else:
                        data[i] = np.sum(contact_lengths)
                        i += 1
            else:
                return None, "Not implemented yet..."
        elif feature in global_features:
            if feature == "total_area":
                data = self.calculate_total_area(valid_cells)
            elif feature == "density":
                data = self.calculate_density(frame, valid_cells, reference)
            elif feature == "type_fraction":
                data = self.calculate_type_fraction(frame, valid_cells, reference)
            elif feature == "number_of_cells":
                data = valid_cells.shape[0]
            elif "neighbors correlation" in feature:
                method = "neighbors_average" if "average" in feature else "neighbors"
                data = self.calculate_neighbors_correlation_function(frame, valid_cells, set_state_by="type",
                                                                     method=method)
        elif feature in spatial_features:
            cells_info = self.get_cells_info(frame)
            relevant_cells = self.get_valid_non_edge_cells(frame, cells_info)
            data = np.zeros((valid_cells.shape[0],))
            msg = ""
            for index in range(valid_cells.shape[0]):
                data[index], current_msg = self.calculate_data_around_a_given_cell(frame, valid_cells.iloc[index],
                                                                                   valid_cells=relevant_cells,
                                                                                   radius=window_radius,
                                                                                   feature=feature,
                                                                                   cells_type='all')
                msg += current_msg
        elif ':' in feature:
            splitted_feature = feature.split(":")
            shape_name = splitted_feature[0]
            shape_feature = splitted_feature[1]
            if shape_name in self.shape_fitting_results[frame - 1]:
                data = self.shape_fitting_results[frame - 1][shape_name][shape_feature]
            else:
                data = valid_cells[feature].to_numpy()
        else:
            data = valid_cells[feature].to_numpy()
        return data, ""

    def calculate_mean_intensity(self,frame, valid_cells, intensity_img):
        labels = self.get_labels(frame)
        props = regionprops_table(labels, intensity_img, properties=('label', 'intensity_mean'))
        valid_cells_labels = valid_cells.index.to_numpy() + 1
        valid_cells_indices= np.intersect1d(props['label'], valid_cells_labels, return_indices=True)[1]
        return props['intensity_mean'][valid_cells_indices]

    @staticmethod
    def match_labels_different_frames(labels_reference_frame, labels_wanted_frame):
        max_label = max(np.max(labels_reference_frame), np.max(labels_wanted_frame))
        # Creating arrays in which element i is the location of the label i in the sorted array
        sorting_indices_reference = -1 * np.ones((max_label + 1,)).astype(int)
        sorting_indices_wanted = -1 * np.ones((max_label + 1,)).astype(int)
        for labels, sorting_indices in zip([labels_reference_frame, labels_wanted_frame],
                                           [sorting_indices_reference, sorting_indices_wanted]):
            sorting_indices_without_missing = np.argsort(labels)
            sorting_indices[labels[sorting_indices_without_missing]] = sorting_indices_without_missing
        location_of_ref_in_wanted = -1 * np.ones((labels_reference_frame.size,)).astype(int)
        location_of_ref_in_wanted[sorting_indices_reference[sorting_indices_reference >= 0]] = sorting_indices_wanted[sorting_indices_reference >= 0]
        return location_of_ref_in_wanted


    def calculate_distance_from_ablation(self, frame, valid_cells):
        ablation_events = self.events.query("type == \"ablation\"")
        ablation_frames = ablation_events.start_frame.values
        nearest_frame = ablation_frames[np.argmin(np.abs(ablation_frames - frame))]
        ablation_events_in_nearest_frame = ablation_events.query("start_frame == %d" % nearest_frame)
        ablation_location_x = ablation_events_in_nearest_frame.start_pos_x.values.astype(float)
        ablation_location_y = ablation_events_in_nearest_frame.start_pos_y.values.astype(float)
        cell_labels = valid_cells.label.to_numpy()
        cell_info_ablation_frame = self.get_cells_info(nearest_frame)
        valid_cells_ablation_frame = self.get_valid_non_edge_cells(nearest_frame, cell_info_ablation_frame)
        cell_labels_ablation_frame = valid_cells_ablation_frame.label.to_numpy()
        indices_ablation_frame = self.match_labels_different_frames(cell_labels, cell_labels_ablation_frame)
        cells_location_x = valid_cells_ablation_frame.iloc[indices_ablation_frame[indices_ablation_frame >= 0]].cx.to_numpy()
        cells_location_y = valid_cells_ablation_frame.iloc[indices_ablation_frame[indices_ablation_frame >= 0]].cy.to_numpy()

        # creating a matrix where distance[i,j] is the distance from cell i to ablation event j
        distance_from_ablations = np.sqrt((ablation_location_x.reshape((1, len(ablation_location_x))) -
                                           cells_location_x.reshape((len(cells_location_x), 1)))**2 +
                                          (ablation_location_y.reshape((1, len(ablation_location_y))) -
                                           cells_location_y.reshape((len(cells_location_y), 1)))**2)
        res = np.empty((valid_cells.shape[0],))
        res[:] = np.nan
        res[indices_ablation_frame >= 0] = np.min(distance_from_ablations, axis=1).flatten()
        return res


    def get_valid_non_edge_cells(self, frame, cells):
        labels = self.get_labels(frame)
        edge_cells_index = self.detect_edge_cells(labels)
        valid_cells = cells.query("valid == 1 and empty_cell == 0")
        return valid_cells[~valid_cells.index.isin(edge_cells_index)]

    def calculate_data_around_a_given_cell(self, frame, cell, valid_cells, radius, feature, cells_type, positive_for_type=True):
        relevant_cells = self.get_cells_inside_a_circle(valid_cells, (cell.cy, cell.cx), radius)
        return self.calculate_spatial_data_for_given_cells(frame, relevant_cells, feature, cells_type, positive_for_type=positive_for_type)

    def calculate_data_around_a_given_point(self, frame, point_x, point_y, valid_cells, radius, feature, cells_type, positive_for_type=True):
        relevant_cells = self.get_cells_inside_a_circle(valid_cells, (point_y, point_x), radius)
        return self.calculate_spatial_data_for_given_cells(frame, relevant_cells, feature, cells_type, positive_for_type=positive_for_type)

    def calculate_spatial_data_for_given_cells(self, frame, relevant_cells, feature, cells_type, positive_for_type=True):
        if feature in self.SPATIAL_FEATURES:
            split_feature = feature.split(" ")
            feature = split_feature[1]
            cells_type = split_feature[0]

        if feature == "density":
            reference = relevant_cells["area"].sum()
        elif feature == "type_fraction":
            reference = relevant_cells.shape[0]
        else:
            reference = 1
        if cells_type != 'all':
            type_index = self.type_name_to_index(cells_type)
            if positive_for_type:
                relevant_cells = relevant_cells.loc[is_positive_for_type(relevant_cells.type.to_numpy(),type_index)]
            else:
                relevant_cells = relevant_cells.loc[~is_positive_for_type(relevant_cells.type.to_numpy(), type_index)]
        # Calculate feature average
        if relevant_cells.shape[0] > 0:
            return self.get_frame_data(frame, feature, relevant_cells,
                                       special_features=self.SPECIAL_FEATURES + self.SPECIAL_X_ONLY_FEATURES,
                                       spatial_features=self.SPATIAL_FEATURES,
                                       global_features=self.GLOBAL_FEATURES,
                                       for_histogram=True, reference=reference)
        elif feature in ["density", "type_fraction"]:
            return 0, ""
        else:
            return None, "No matching cells"


    def calculate_spatial_data(self, frame, window_radius, step_size, feature, cells_type='all', positive_for_type=True):
        labels = self.get_labels(frame)
        cells_info = self.get_cells_info(frame)
        res = np.zeros(labels.shape)
        valid_cells = self.get_valid_non_edge_cells(frame, cells_info)
        for y in range(step_size//2, res.shape[0], step_size):
            for x in range(step_size//2, res.shape[1], step_size):
                data, err_msg = self.calculate_data_around_a_given_point(frame, x, y,
                                                                         valid_cells, window_radius, feature,
                                                                         cells_type, positive_for_type=positive_for_type)
                if err_msg:
                    return None, err_msg
                if hasattr(data, "__len__"):
                    if len(data) > 0:
                        data = np.average(data)
                    else:
                        data = 0
                # fill up result
                res[y - step_size//2:y + step_size//2, x-step_size//2:x+step_size//2] = data
        return res, ""



    @staticmethod
    def get_cells_inside_a_circle(cells, center, radius):
        center_x = center[1]
        center_y = center[0]
        return cells.query("(cx - %f)**2 + (cy - %f)**2 < %f" % (center_x, center_y, radius**2))


    def plot_single_frame_data(self, frame, x_feature, y_feature, ax, cells_type='all', positive_for_type=True, intensity_image=None):
        cell_info = self.get_cells_info(frame)
        type_index = self.type_name_to_index(cells_type)
        y_data = None
        if cell_info is None:
            return None, "No frame data is available"
        if cells_type == "all":
            valid_cells = self.get_valid_non_edge_cells(frame, cell_info)
        else:
            if positive_for_type:
                cells_from_right_type = cell_info.loc[is_positive_for_type(cell_info.type.to_numpy(), type_index)]
            else:
                cells_from_right_type = cell_info.loc[~is_positive_for_type(cell_info.type.to_numpy(), type_index)]
            valid_cells = self.get_valid_non_edge_cells(frame, cells_from_right_type)
        plotted = False
        x_data, msg = self.get_frame_data(frame, x_feature, valid_cells,
                                          self.SPECIAL_FEATURES + self.SPECIAL_X_ONLY_FEATURES,
                                          intensity_img=intensity_image)
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
                                          self.SPECIAL_FEATURES + self.SPECIAL_Y_ONLY_FEATURES)
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

    def plot_spatial_map_over_time(self, frames, feature, window_radius, window_step, ax, cells_type='all', positive_for_type=True):
        maps_list = []
        valid_frames = []
        for frame in frames:
            if self.is_frame_valid(frame):
                map, msg = self.calculate_spatial_data(frame, window_radius, window_step, feature, cells_type=cells_type, positive_for_type=positive_for_type)
                valid_frames.append(frame)
            maps_list.append(map)
        maps_list = np.dstack(maps_list)
        time = np.array(valid_frames)/4
        for x in maps_list.shape[0]:
            for y in maps_list.shape[1]:
                location_x = np.round(x*window_step + window_radius/2)
                location_y = np.round(y*window_step + window_radius/2)
                ax.plot(valid_frames, maps_list[x, y, :], label="x=%d um, y=%d um" % (location_x, location_y))
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Neighbors Correlation")

    def plot_spatial_map(self, frame, feature, window_radius, window_step, ax, cells_type='all', positive_for_type=True, vmin=None, vmax=None):
        map, msg = self.calculate_spatial_data(frame, window_radius, window_step, feature, cells_type=cells_type, positive_for_type=positive_for_type)
        labels = self.get_labels(frame)
        palette = copy.copy(colormap.RdBu)
        palette.set_bad('k')
        palette.set_under('k')
        if 'fraction' in feature:
            vmin = 0
            vmax = 1
        if vmin is None:
            vmin = np.min(map[map > 0])
        if vmax is None:
            vmax = np.max(map[map > 0])
        map_masked = np.ma.masked_where(labels == 0, map)
        im = ax.imshow(map_masked, cmap=palette, vmin=vmin, vmax=vmax)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(feature, rotation=270)
        return map, msg

    def plot_compare_frames_data(self, frames, feature, ax, cells_type='all', positive_for_type=True):
        type_index = self.type_name_to_index(cells_type)
        data = []
        err = []
        n_results = []
        data_frames = []
        for frame in frames:
            if not self.is_frame_valid(frame):
                continue
            data_frames.append(frame)
            cell_info = self.get_cells_info(frame)
            if cell_info is None:
                return None, "No frame data is available for frame %d" % frame
            if cells_type == "all":
                valid_cells = self.get_valid_non_edge_cells(frame, cell_info)
            else:
                if positive_for_type:
                    cells_from_right_type = cell_info.loc[is_positive_for_type(cell_info.type.to_numpy(), type_index)]
                else:
                    cells_from_right_type = cell_info.loc[~is_positive_for_type(cell_info.type.to_numpy(), type_index)]
                valid_cells = self.get_valid_non_edge_cells(frame, cells_from_right_type)
            raw_data, msg = self.get_frame_data(frame, feature, valid_cells,
                                              self.SPECIAL_FEATURES + self.SPECIAL_X_ONLY_FEATURES,
                                              global_features=self.GLOBAL_FEATURES,
                                              for_histogram=True)
            if raw_data is None:
                return None, msg
            bar_plot = True
            if isinstance(raw_data, tuple): # from shape fitting
                data.append(raw_data[0])
                err.append(raw_data[1])
                n_results.append(1)
                bar_plot = False
            elif hasattr(raw_data, "__len__") and len(raw_data) > 1:
                data.append(np.average(raw_data))
                err.append(np.std(raw_data)/np.sqrt(np.size(raw_data)))
                n_results.append(np.size(raw_data))
            else:
                data.append(raw_data)
                err.append(0)
                n_results.append(1)
                bar_plot = False
        if not bar_plot:
            ax.errorbar(data_frames, data, yerr = err, fmt="*")
        else:
            x_pos = np.arange(len(data_frames))
            x_labels = ["frame %d (N = %d)" % (f, n) for f,n in zip(data_frames, n_results)]
            ax.bar(x_pos, data, yerr=err, align='center', alpha=0.5, ecolor='black', capsize=10)
            ax.set_ylabel(feature)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)
            ax.yaxis.grid(True)
        title = "%s for different frames" % feature
        if cells_type != 'all':
            title += " for %s only" % cells_type
        ax.set_title(title)
        return pd.DataFrame({"Frame": data_frames, feature + " average": data, feature + " se": err, "N": n_results}), ""

    def plot_overall_statistics(self, frame, x_feature, y_feature, ax, intensity_img=None,
                                x_cells_type="HC", x_positive_for_type=False,  y_cells_type="HC", y_positive_for_type=False,
                                x_radius=0, y_radius=0):
        if ("intensity" not in x_feature) and ((y_feature is None) or ("intensity" not in y_feature)):
            intensity_img = None
        cells_info = self.get_cells_info(frame)
        valid_cells = self.get_valid_non_edge_cells(frame, cells_info)
        if x_cells_type != "all":
            x_type_index = self.type_name_to_index(x_cells_type)
            if x_positive_for_type:
                x_valid_cells = valid_cells.loc[is_positive_for_type(valid_cells.type.to_numpy(), x_type_index)]
            else:
                x_valid_cells = valid_cells.loc[~is_positive_for_type(valid_cells.type.to_numpy(), x_type_index)]
        else:
            x_valid_cells = valid_cells
        x_data, x_msg = self.get_frame_data(frame, x_feature, x_valid_cells,
                                          special_features=self.SPECIAL_FEATURES,
                                          global_features=self.GLOBAL_FEATURES,
                                          spatial_features=self.SPATIAL_FEATURES,
                                          for_histogram=False,
                                          intensity_img=intensity_img,
                                          window_radius=x_radius)
        if y_feature is not None:
            if y_cells_type != "all":
                y_type_index = self.type_name_to_index(y_cells_type)
                if y_positive_for_type:
                    y_valid_cells = valid_cells.loc[is_positive_for_type(valid_cells.type.to_numpy(), y_type_index)]
                else:
                    y_valid_cells = valid_cells.loc[~is_positive_for_type(valid_cells.type.to_numpy(), y_type_index)]
            else:
                y_valid_cells = valid_cells
            y_data, y_msg = self.get_frame_data(frame, y_feature, y_valid_cells,
                                                special_features=self.SPECIAL_FEATURES,
                                                global_features=self.GLOBAL_FEATURES,
                                                spatial_features=self.SPATIAL_FEATURES,
                                                for_histogram=False,
                                                intensity_img=intensity_img,
                                                window_radius=y_radius)
        if y_feature is None:
            ax.hist(x_data[~np.isnan(x_data)])
            ax.set_xlabel(x_feature)
            ax.set_ylabel('Number of cells')
            title = "%s histogram for frame %d" % (x_feature, frame)
            res = pd.DataFrame({"event type": "overall", x_feature: x_data})
        else:
            histogram = ax.hist2d(x_data[~np.isnan(x_data)], y_data[~np.isnan(y_data)])
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            cbar = ax.get_figure().colorbar(histogram[3], ax=ax)
            title = "%s, %s histogram for frame %d" % (x_feature, y_feature, frame)
            cbar.set_label('Number of events', rotation=270)
            res = pd.DataFrame({"event type": "overall", x_feature: x_data, y_feature:y_data})
        ax.set_title(title)
        return res, ""

    def plot_event_statistics(self, event_type, x_feature, x_radius, y_feature, y_radius, ax, intensity_images):
        if "significant_frame" not in self.events.columns:
            self.find_events_frame()
        event_data = self.events.query("type == \"%s\"" % event_type)
        if event_data.shape[0] < 1:
            return None, "No matching events of type %s" % event_type
        else:
            if x_feature in self.SPECIAL_EVENT_STATISTICS_FEATURES:
                if x_feature == "timing histogram":
                    x_data = event_data.significant_frame.to_numpy().astype("float")
                elif x_feature == "spatio-temporal correlation":
                    x_data = self.calculate_events_correlation_function(y_radius, x_radius, event_type)
                    im = ax.matshow(np.flipud(x_data.T), cmap=plt.cm.RdBu,
                                    extent=[0, x_data.shape[0] * x_radius * 15, 0,
                                            x_data.shape[1] * y_radius * 0.1], aspect='auto',
                                    vmin=0, vmax=np.max(x_data[x_data < x_data[0,0]]))
                    plt.colorbar(im)
                    ax.set_xlabel("Time (minutes)")
                    ax.set_ylabel("Distance (microns)")
                    time_axis = np.arange(start=0, stop=x_data.shape[0] * x_radius * 15, step=x_radius*15)
                    distance_axis = np.arange(start=0, stop=x_data.shape[1] * y_radius * 0.1, step=y_radius*0.1)
                    msg = ""
                    res_dict = {"event type": event_type, "distance axis": distance_axis,
                                        "time axis": time_axis, "correlation": x_data}
                    return res_dict, msg
                x_data_calculated = True
            else:
                x_data = np.zeros(event_data.shape[0])
                x_data_calculated = False
            if y_feature is not None:
                if y_feature in self.SPECIAL_EVENT_STATISTICS_FEATURES:
                    if y_feature == "timing histogram":
                        y_data = event_data.significant_frame.to_numpy().astype("float")
                    y_data_calculated = True
                else:
                    y_data = np.zeros(event_data.shape[0])
                    y_data_calculated = False
            index = 0
            msg = ""
            for event_index, event in event_data.iterrows():
                cell_id = event.cell_id
                frame = int(event.significant_frame)
                if ("intensity" not in x_feature) and ((y_feature is None) or ("intensity" not in y_feature)):
                    intensity_img = None
                else:
                    intensity_img = intensity_images[frame - 1]
                if not x_data_calculated:
                    current_x_data, current_msg = self.get_single_cell_data(cell_id, [frame], x_feature, [intensity_img],
                                                                         window_radius=x_radius)
                    x_data[index] = current_x_data
                    msg += current_msg
                if y_feature is not None and not y_data_calculated:
                        current_y_data, current_msg = self.get_single_cell_data(cell_id, [frame], y_feature, [intensity_img],
                                                                             window_radius=y_radius)
                        y_data[index] = current_y_data
                        msg += current_msg
                index += 1
            if y_feature is None:
                ax.hist(x_data[~np.isnan(x_data)])
                ax.set_xlabel(x_feature)
                ax.set_ylabel('Number of events')
                title = "%s histogram for %s" % (x_feature, event_type)
                res = pd.DataFrame({"event type": event_type, x_feature: x_data})
            else:
                valid = np.logical_and(~np.isnan(x_data), ~np.isnan(y_data))
                histogram = ax.hist2d(x_data[valid], y_data[valid])
                ax.set_xlabel(x_feature)
                ax.set_ylabel(y_feature)
                cbar = ax.get_figure().colorbar(histogram[3], ax=ax)
                title = "%s, %s histogram for %s" % (x_feature, y_feature, event_type)
                cbar.set_label('Number of events', rotation=270)
                res = pd.DataFrame({"event type": event_type, x_feature: x_data, y_feature:y_data})
            ax.set_title(title)
        return res, msg

    def split_into_promoted_and_normal_differentiation(self, threshold):
        fig, ax = plt.subplots()
        res, msg = self.plot_event_statistics("differentiation", "Distance from ablation", 0, None, 0, ax, None)
        differentiation_indices = self.events.query("type == \"differentiation\"").index.to_numpy()
        near_ablation = res["Distance from ablation"].to_numpy() < threshold
        self.events.loc[differentiation_indices[near_ablation], "type"] = "promoted differentiation"
        return 0

    @staticmethod
    def calculate_cells_roundness(cells):
        return cells.eval("4 * %f * area / (perimeter ** 2)" % np.pi).to_numpy()

    @staticmethod
    def calculate_cells_shape_index(cells):
        return cells.eval("perimeter/(area**(1/2))").to_numpy()

    @staticmethod
    def calculate_total_area(cells):
        return np.sum(cells.area.to_numpy())

    def calculate_density(self, frame, relevant_cells, reference_area=None):
        cells_info = self.get_cells_info(frame)
        if cells_info is None:
            return -1
        if reference_area is None:
            all_cells = cells_info.query("empty_cell == 0")
            reference_area = self.calculate_total_area(all_cells)
        if reference_area > 0:
            return relevant_cells.shape[0]/reference_area
        else:
            return 0

    def calculate_type_fraction(self, frame, relevant_cells, reference_cell_num=None):
        cells_info = self.get_cells_info(frame)
        if cells_info is None:
            return -1
        if reference_cell_num is None:
            all_cells = cells_info.query("valid == 1 and empty_cell == 0")
            reference_cell_num = all_cells.shape[0]
        if reference_cell_num > 0:
            return relevant_cells.shape[0] / reference_cell_num
        else:
            return 0

    def calculate_n_neighbors_from_type(self, frame, cells, cell_type='same', positive_for_type=True, second_neighbors=False):
        cell_info = self.get_cells_info(frame)
        if cell_info is None:
            return None
        neighbors_from_type = np.zeros((cells.shape[0],))
        if second_neighbors:
            second_type = 'all' if cell_type == 'same' else cell_type
            second_order_neighbors = self.find_second_order_neighbors(frame, cells, second_type, positive_for_type=positive_for_type)
        index = 0
        for i, row in cells.iterrows():
            if cell_type == 'same':
                look_for_type = row.type
            else:
                look_for_type = cell_type
            if second_neighbors:
                neighbors = np.array(list(second_order_neighbors[index]))
            else:
                neighbors = np.array(list(row.neighbors))
            if (cell_type != 'all' and (not second_neighbors)) or cell_type == 'same':
                neighbors_data = cell_info.loc[neighbors - 1, ["valid", "type", "empty_cell"]]
                type_index = self.type_name_to_index(look_for_type)
                if positive_for_type:
                    valid_neighbors_from_the_right_type = neighbors_data.loc[
                        is_positive_for_type(neighbors_data.type.to_numpy(), type_index)].query(
                        "valid == 1 and empty_cell == 0")
                else:
                    valid_neighbors_from_the_right_type = neighbors_data.loc[
                        ~is_positive_for_type(neighbors_data.type.to_numpy(), type_index)].query(
                        "valid == 1 and empty_cell == 0")
                neighbors_from_type[index] = valid_neighbors_from_the_right_type.shape[0]
            else:
                neighbors_from_type[index] = neighbors.size
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

    def calculate_contact_length(self, frame, cell_info, max_filtered_labels, min_filtered_labels, cell_type='all', positive_for_type=True):
        cells_info = self.get_cells_info(frame)
        cell_label = cells_info.query("label == %d and valid == 1 and empty_cell == 0" % cell_info.label).index.to_numpy() + 1
        region_first_row = int(max(0, cell_info.bounding_box_min_row - 2))
        region_last_row = int(cell_info.bounding_box_max_row + 2)
        region_first_col = int(max(0, cell_info.bounding_box_min_col - 2))
        region_last_col = int(cell_info.bounding_box_max_col + 2)
        max_filtered_region = max_filtered_labels[region_first_row:region_last_row, region_first_col:region_last_col]
        min_filtered_region = min_filtered_labels[region_first_row:region_last_row, region_first_col:region_last_col]
        neighbor_labels = np.array(list(cell_info.neighbors.copy()))
        if cell_type == 'all':
            neighbors_from_the_right_type = (cells_info.valid[np.array(neighbor_labels) - 1] == 1).to_numpy()
        else:
            type_index = self.type_name_to_index(cell_type)
            neighbors_from_the_right_type = is_positive_for_type(cells_info.type[np.array(neighbor_labels) - 1], type_index).to_numpy()
            if not positive_for_type:
                neighbors_from_the_right_type = ~neighbors_from_the_right_type
        neighbor_labels = neighbor_labels[neighbors_from_the_right_type]
        contact_length = []
        for neighbor_label in neighbor_labels:
            max_label = max(cell_label, neighbor_label)
            min_label = min(cell_label, neighbor_label)
            contact_length.append(np.sum(np.logical_and(max_filtered_region == max_label, min_filtered_region == min_label)).astype(int))
        return neighbor_labels, contact_length

    def track_cells(self, initial_frame=1, final_frame=-1, images=None, image_in_memory=False):
        iter = self.track_cells_iterator_with_trackpy(initial_frame, final_frame, images, image_in_memory)
        last_frame = initial_frame
        for frame in iter:
            last_frame = frame
        return last_frame

    def track_cells_iterator_with_trackpy(self, initial_frame=1, final_frame=-1, images=None, image_in_memory=False):
        if final_frame == -1:
            final_frame = self.number_of_frames
        use_existing_drifts = (self.drifts > 0).any()

        def cells_info_iterator(tissue):
            previous_frame = 0
            update_next_drift = False
            overall_drift_x = 0
            overall_drift_y = 0
            index = 0
            for frame in range(initial_frame, final_frame + 1):
                if tissue.valid_frames[frame - 1] == 0:
                    if not np.isnan(tissue.drifts[frame - 1, 0]):
                        tissue.drifts[frame - 1, :] = np.nan
                        update_next_drift = True
                    continue
                cells_info = tissue.get_cells_info(frame)
                if cells_info is None:
                    continue
                frame_data = cells_info.query("valid == 1 and empty_cell == 0").copy(deep=True)
                frame_data['frame'] = frame
                frame_data['frame_index'] = index
                index += 1
                # Fix drifts
                if frame > initial_frame:
                    if use_existing_drifts and not update_next_drift:
                        overall_drift_x += tissue.drifts[frame - 1, 1]
                        frame_data.loc[:,"cx"] = frame_data["cx"] + overall_drift_x
                        overall_drift_y += tissue.drifts[frame - 1, 0]
                        frame_data.loc[:,"cy"] = frame_data["cy"] + overall_drift_y
                    else:
                        shift_y, shift_x = tissue.update_drift(frame, previous_frame, images=images,
                                                               image_in_memory=image_in_memory)
                        overall_drift_x += shift_x
                        frame_data.loc[:,"cx"] = frame_data["cx"] + overall_drift_x
                        overall_drift_y += shift_y
                        frame_data.loc[:,"cy"] = frame_data["cy"] + overall_drift_y
                previous_frame = frame
                yield frame_data

        tracking_iter = trackpy.link_df_iter(cells_info_iterator(self), search_range=100, adaptive_stop=10,
                                             pos_columns=["cy", "cx", "area"], t_column="frame_index", memory=3,
                                             neighbor_strategy='BTree', dist_func=self.tracking_dist_func)
        for frame_data in tracking_iter:
            frame = frame_data.frame.values[0]
            cells_info = self.get_cells_info(frame)
            cells_info.loc[frame_data.index, "label"] = frame_data["particle"] + 1
            yield frame
        return 0

    @staticmethod
    def tracking_dist_func(first, second):
        return np.sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2 + 0.5 * (
                    np.sqrt(first[2]) - np.sqrt(second[2])) ** 2)

    @staticmethod
    def calculate_refine_drift(previous_image, current_image,
                               course_shift_x, course_shift_y):
        # Refine drift prediction by phase cross correlation
        rounded_shift_x = int(np.floor(course_shift_x))
        rounded_shift_y = int(np.floor(course_shift_y))
        if rounded_shift_x > 0:
            if rounded_shift_y > 0:
                previous_img = previous_image[rounded_shift_x:, rounded_shift_y:]
                current_img = current_image[:-rounded_shift_x, :-rounded_shift_y]
            elif rounded_shift_y < 0:
                previous_img = previous_image[rounded_shift_x:, :rounded_shift_y]
                current_img = current_image[:-rounded_shift_x, -rounded_shift_y:]
            else:
                previous_img = previous_image[rounded_shift_x:, :]
                current_img = current_image[:-rounded_shift_x, :]
        elif rounded_shift_x < 0:
            if rounded_shift_y > 0:
                previous_img = previous_image[:rounded_shift_x, rounded_shift_y:]
                current_img = current_image[-rounded_shift_x:, :-rounded_shift_y]
            elif rounded_shift_y < 0:
                previous_img = previous_image[:rounded_shift_x, :rounded_shift_y]
                current_img = current_image[-rounded_shift_x:, -rounded_shift_y:]
            else:
                previous_img = previous_image[:rounded_shift_x, :]
                current_img = current_image[-rounded_shift_x:, :]
        else:
            if rounded_shift_y > 0:
                previous_img = previous_image[:, rounded_shift_y:]
                current_img = current_image[:, :-rounded_shift_y]
            elif rounded_shift_y < 0:
                previous_img = previous_image[:, :rounded_shift_y]
                current_img = current_image[:, -rounded_shift_y:]
            else:
                previous_img = previous_image[:, :]
                current_img = current_image[:, :]
        refined_shift, error, diffphase = phase_cross_correlation(previous_img, current_img,
                                                                  upsample_factor=100)
        shift_x = rounded_shift_x + refined_shift[-2]
        shift_y = rounded_shift_y + refined_shift[-1]
        return shift_x, shift_y

    def update_drift(self, frame, previous_frame, images=None, image_in_memory=False):
        if self.stage_locations is not None:
            shift = (self.stage_locations.loc[frame - 1, ["z", "y", "x"]].to_numpy() -
                     self.stage_locations.loc[previous_frame - 1, ["z", "y", "x"]].to_numpy()) / \
                     self.stage_locations.loc[
                        frame - 1, ["physical_size_z", "physical_size_y", "physical_size_x"]].to_numpy()
        else:
            shift = (0, 0)
        shift_x = shift[-2]  # x/y are swapped between stage location and image
        shift_y = shift[-1]
        if images is not None:
            # Refine drift prediction by phase cross correlation
            rounded_shift_x = int(np.floor(shift_x))
            rounded_shift_y = int(np.floor(shift_y))
            if rounded_shift_x > 0:
                if rounded_shift_y > 0:
                    previous_img = images[previous_frame - 1, rounded_shift_x:, rounded_shift_y:]
                    current_img = images[frame - 1, :-rounded_shift_x, :-rounded_shift_y]
                elif rounded_shift_y < 0:
                    previous_img = images[previous_frame - 1, rounded_shift_x:, :rounded_shift_y]
                    current_img = images[frame - 1, :-rounded_shift_x, -rounded_shift_y:]
                else:
                    previous_img = images[previous_frame - 1, rounded_shift_x:, :]
                    current_img = images[frame - 1, :-rounded_shift_x, :]
            elif rounded_shift_x < 0:
                if rounded_shift_y > 0:
                    previous_img = images[previous_frame - 1, :rounded_shift_x, rounded_shift_y:]
                    current_img = images[frame - 1, -rounded_shift_x:, :-rounded_shift_y]
                elif rounded_shift_y < 0:
                    previous_img = images[previous_frame - 1, :rounded_shift_x, :rounded_shift_y]
                    current_img = images[frame - 1, -rounded_shift_x:, -rounded_shift_y:]
                else:
                    previous_img = images[previous_frame - 1, :rounded_shift_x, :]
                    current_img = images[frame - 1, -rounded_shift_x:, :]
            else:
                if rounded_shift_y > 0:
                    previous_img = images[previous_frame - 1, :, rounded_shift_y:]
                    current_img = images[frame - 1, :, :-rounded_shift_y]
                elif rounded_shift_y < 0:
                    previous_img = images[previous_frame - 1, :, :rounded_shift_y]
                    current_img = images[frame - 1, :, -rounded_shift_y:]
                else:
                    previous_img = images[previous_frame - 1, :, :]
                    current_img = images[frame - 1, :, :]
            if not image_in_memory:
                previous_img = previous_img.compute()
                current_img = current_img.compute()
            refined_shift, error, diffphase = phase_cross_correlation(previous_img, current_img,
                                                                        upsample_factor=100)
            shift_x = rounded_shift_x + refined_shift[-2]
            shift_y = rounded_shift_y + refined_shift[-1]
        self.drifts[frame - 1, 0] = shift_y
        self.drifts[frame - 1, 1] = shift_x
        return shift_y, shift_x

    def track_cells_iterator(self, initial_frame=1, final_frame=-1, images=None, image_in_memory=False, use_piv=False):
        if final_frame == -1:
            final_frame = self.number_of_frames
        cells_info = self.get_cells_info(initial_frame)
        if cells_info is None:
            return 0
        unlabeled_cells = (cells_info.label.to_numpy() == 0)
        last_used_label = cells_info.label.max()
        cells_info.loc[unlabeled_cells, "label"] = np.arange(last_used_label + 1,
                                                            last_used_label + np.sum(unlabeled_cells.astype(int)) + 1)
        cx_previous_frame = np.copy(cells_info.cx.to_numpy())
        cy_previous_frame = np.copy(cells_info.cy.to_numpy())
        labels_previous_frame = cells_info.label.to_numpy()
        empty_cells_previous_frame = cells_info.empty_cell.to_numpy()
        previous_frame = initial_frame
        self.cells_number = max(self.cells_number, cells_info.label.max())
        use_existing_drifts = (self.drifts > 0).any()
        update_next_drift = False
        for frame in range(initial_frame + 1, final_frame + 1):
            if self.valid_frames[frame - 1] == 0:
                if not np.isnan(self.drifts[frame - 1,0]):
                    self.drifts[frame - 1, :] = np.nan
                    update_next_drift = True
                continue
            if use_piv and images is not None:
                last_image = images[previous_frame - 1]
                current_image = images[frame - 1]
                if not image_in_memory:
                    last_image = last_image.compute()
                    current_image = current_image.compute()
                piv_x, piv_y = optical_flow_tvl1(last_image, current_image)
                drift_map_x = piv_x[np.round(cx_previous_frame).astype(int), np.round(cy_previous_frame).astype(int)]
                drift_map_y = piv_y[np.round(cx_previous_frame).astype(int), np.round(cy_previous_frame).astype(int)]
                cx_previous_frame -= drift_map_x
                cy_previous_frame -= drift_map_y
            elif use_existing_drifts and not update_next_drift:
                cx_previous_frame -= self.drifts[frame - 1, 1]
                cy_previous_frame -= self.drifts[frame - 1, 0]
            else:
                shift_y, shift_x = self.update_drift(frame, previous_frame, images=images, image_in_memory=image_in_memory)
                cx_previous_frame -= shift_x
                cy_previous_frame -= shift_y

            cells_info = self.get_cells_info(frame)
            labels = maximum_filter(self.get_labels(frame), (3, 3), mode='constant')
            if cells_info is None or labels is None:
                continue
            cells_info.loc[:, "label"] = 0
            indices_in_current_frame = -1 * np.ones(cy_previous_frame.shape)
            y_locations = np.round(cy_previous_frame).astype(int)
            x_locations = np.round(cx_previous_frame).astype(int)
            valid_locations = np.logical_and(np.logical_and(np.logical_and(0 <= y_locations, y_locations < labels.shape[0]),
                              np.logical_and(0 <= x_locations, x_locations < labels.shape[1])), empty_cells_previous_frame == 0)
            indices_in_current_frame[valid_locations] = labels[np.round(cy_previous_frame[valid_locations]).astype(int),
                                              np.round(cx_previous_frame[valid_locations]).astype(int)] - 1
            labels_previous_frame = labels_previous_frame[indices_in_current_frame >= 0]
            indices_in_current_frame = indices_in_current_frame[indices_in_current_frame >= 0]
            _, location_of_unique_labels= np.unique(labels_previous_frame, return_index=True)
            indices_in_current_frame = indices_in_current_frame[location_of_unique_labels]
            labels_previous_frame = labels_previous_frame[location_of_unique_labels]
            _, location_of_unique_indices = np.unique(indices_in_current_frame, return_index=True)
            indices_in_current_frame = indices_in_current_frame[location_of_unique_indices]
            labels_previous_frame = labels_previous_frame[location_of_unique_indices]
            cells_info.loc[indices_in_current_frame.astype("int"), "label"] = labels_previous_frame
            unlabeled_cells = (cells_info.label.to_numpy() == 0)
            last_used_label = cells_info.label.max()
            cells_info.loc[unlabeled_cells, "label"] = np.arange(last_used_label + 1,
                                                                last_used_label + np.sum(
                                                                    unlabeled_cells.astype(int)) + 1)
            self.cells_number = max(self.cells_number, cells_info.label.max())
            cx_previous_frame = np.copy(cells_info.cx.to_numpy())
            cy_previous_frame = np.copy(cells_info.cy.to_numpy())
            labels_previous_frame = cells_info.label.to_numpy()
            empty_cells_previous_frame = cells_info.empty_cell.to_numpy()
            previous_frame = frame
            yield frame
        return 0

    def fix_one_frame_tracking_using_local_drifts(self, start_frame, images, step_size=100, window_size=500,
                                                  image_in_memory=False, shift_x=0, shift_y=0):

        # find next valid frame
        next_frame = -1
        for frame in range(start_frame + 1, self.number_of_frames):
            if self.valid_frames[frame - 1] == 1:
                next_frame = frame
                break
        if next_frame < 0:
            return 0
        if shift_x != 0 or shift_y != 0:
            shift = (shift_y, shift_x)
        elif self.stage_locations is not None:
            shift = (self.stage_locations.loc[next_frame - 1, ["z", "y", "x"]].to_numpy() -
                     self.stage_locations.loc[start_frame - 1, ["z", "y", "x"]].to_numpy()) / \
                    self.stage_locations.loc[
                        frame - 1, ["physical_size_z", "physical_size_y", "physical_size_x"]].to_numpy()
        else:
            shift = (0, 0)
        initial_shift_x = shift[-2]  # x/y are swapped between stage location and image
        initial_shift_y = shift[-1]

        first_image = images[start_frame - 1]
        second_image = images[next_frame - 1]
        first_cell_info = self.get_cells_info(start_frame).query("valid == 1 and empty_cell == 0")
        first_cx = np.copy(first_cell_info.cx.to_numpy())
        first_cy = np.copy(first_cell_info.cy.to_numpy())
        if not image_in_memory:
            first_image = first_image.compute()
            second_image = second_image.compute()
        local_shifts_x = np.zeros(first_image.shape)
        local_shifts_y = np.zeros(first_image.shape)
        shifts_counter = np.zeros(first_image.shape)
        for initial_row in range(0, first_image.shape[0] - window_size, step_size):
            for initial_col in range(0, first_image.shape[1] - window_size, step_size):
                if initial_row + step_size + window_size > first_image.shape[0]:
                    final_row = first_image.shape[0]
                else:
                    final_row = initial_row+window_size
                if initial_col + step_size + window_size > first_image.shape[1]:
                    final_col = first_image.shape[1]
                else:
                    final_col = initial_col + window_size
                local_first_image = first_image[initial_row:final_row, initial_col:final_col]
                local_second_image = second_image[initial_row:final_row, initial_col:final_col]
                shift_x, shift_y = self.calculate_refine_drift(local_first_image, local_second_image, initial_shift_x, initial_shift_y)
                local_shifts_x[initial_row:final_row, initial_col:final_col] += shift_x
                local_shifts_y[initial_row:final_row, initial_col:final_col] += shift_y
                shifts_counter[initial_row:final_row, initial_col:final_col] += 1
        local_shifts_x = local_shifts_x / shifts_counter
        local_shifts_y = local_shifts_y / shifts_counter
        drift_map_x = local_shifts_x[np.round(first_cx).astype(int), np.round(first_cy).astype(int)]
        drift_map_y = local_shifts_y[np.round(first_cx).astype(int), np.round(first_cy).astype(int)]
        first_cx -= drift_map_x
        first_cy -= drift_map_y
        first_frame_linking_info = pd.DataFrame({"cx": first_cx, "cy": first_cy,
                                                 "area": np.copy(first_cell_info.area.to_numpy()),
                                                 "frame_index": np.zeros(first_cx.shape),
                                                 "label": np.copy(first_cell_info.label.to_numpy())}, index=first_cell_info.index,)
        second_cell_info = self.get_cells_info(next_frame).query("valid == 1 and empty_cell == 0")
        second_frame_linking_info = pd.DataFrame({"cx": second_cell_info.cx.to_numpy(), "cy": second_cell_info.cy.to_numpy(),
                                                  "area": np.copy(second_cell_info.area.to_numpy()),
                                                  "frame_index": np.ones((second_cell_info.shape[0],)),
                                                  "label": np.copy(second_cell_info.label.to_numpy())}, index=second_cell_info.index)
        link_info = trackpy.link(pd.concat([first_frame_linking_info, second_frame_linking_info]), search_range=100,
                                 adaptive_stop=10, pos_columns=["cy", "cx", "area"], t_column="frame_index", memory=0,
                                 neighbor_strategy='BTree', dist_func=self.tracking_dist_func)
        # Re-linking the cells
        first_frame_linking_info = link_info.query("frame_index == 0").loc[:, ["label", "particle"]]
        second_frame_linking_info = link_info.query("frame_index == 1").loc[:, ["label", "particle"]]
        first_frame_labels = first_frame_linking_info.label.to_numpy()
        second_frame_particle_numbers = second_frame_linking_info.particle.to_numpy()
        # creating a LUT for the second frame
        old_labels = second_frame_linking_info.label.to_numpy()
        new_labels = np.copy(old_labels)
        # There are 4 groups that should be handled -
        # 1. Unliked cells that appear only in the second frame cells that were unlinked and should stay the same -
        # These were already handled since the old labels were copied
        # 2. Cells that where linked in the new tracking should get their updated label
        linked_labels = second_frame_particle_numbers < first_frame_labels.size
        new_labels[linked_labels] = first_frame_labels[second_frame_particle_numbers[linked_labels]]
        # 3. Unlinked cells which has a label that exist in the first frame - i.e they were linked before but now
        # they're not. These cells should get a new label
        unlinked_recurring_labels = np.logical_and(~linked_labels, np.isin(new_labels, first_frame_labels))
        first_free_label = max(np.max(first_frame_labels), np.max(new_labels)) + 1
        number_of_new_labels = np.sum(unlinked_recurring_labels.astype(int))
        new_labels[unlinked_recurring_labels] = np.arange(first_free_label, first_free_label + number_of_new_labels)
        # At this point we can assign the new labels to the second frame
        assigning_indices = second_frame_linking_info.index.to_numpy()
        self.cells_info.loc[assigning_indices, "label"] = new_labels
        # But for subsequent frames we will also need to handle another group:
        # 4. Unliked cells that do not appear in the second frame - should have an unchanged label since they
        # might skip a frame
        skip_labels = first_frame_labels[np.logical_and(~np.isin(first_frame_labels, old_labels, assume_unique=True),
                                                        ~np.isin(first_frame_labels, new_labels, assume_unique=True))]
        old_labels = np.hstack([old_labels, skip_labels])
        new_labels = np.hstack([new_labels, skip_labels])
        # Updating labels in all subsequent frames
        for frame in range(next_frame + 1, self.number_of_frames):
            if self.is_frame_valid(frame):
                cells_info = self.get_cells_info(frame).query("valid == 1 and empty_cell == 0")
                current_frame_labels = cells_info.label.to_numpy()
                # First we need to update the LUT, There are 3 kinds of labels -
                # 1. Labels that are already in the LUT keys - no need to do anything
                # 2. Labels that are not in the LUT keys or values - should be skipped
                in_keys = np.isin(current_frame_labels, old_labels, assume_unique=True)
                in_values = np.isin(current_frame_labels, new_labels, assume_unique=True)
                skip_labels = np.logical_and(~in_keys, ~in_values)
                old_labels = np.hstack([old_labels, current_frame_labels[skip_labels]])
                new_labels = np.hstack([new_labels, current_frame_labels[skip_labels]])
                # 3. Labels that are not in the LUT keys but are already in its values - should get a new label
                needs_new_label = np.logical_and(~in_keys, in_values)
                first_free_label = max(np.max(old_labels), np.max(new_labels)) + 1
                number_of_new_labels = np.sum(needs_new_label.astype(int))
                old_labels = np.hstack([old_labels, current_frame_labels[needs_new_label]])
                new_labels = np.hstack([new_labels, np.arange(first_free_label, first_free_label + number_of_new_labels)])
                # Now that the LUT has been updates we can assign the new values
                # We first take only the keys wee need
                relevant_indices = np.isin(old_labels, current_frame_labels, assume_unique=True)
                relevant_old_labels = old_labels[relevant_indices]
                relevant_new_labels = new_labels[relevant_indices]
                # sorting relevant LUT
                keys_sorting_indices = np.argsort(relevant_old_labels)
                # Assigning values
                labels_sorting_indices = np.argsort(current_frame_labels)
                assigning_indices = cells_info.index.to_numpy()
                self.cells_info.loc[assigning_indices[labels_sorting_indices], "label"] = relevant_new_labels[keys_sorting_indices]
        return 0

    def fix_cell_label(self, frame, position, new_label):
        if new_label <= 0:
            return 0
        labels = self.get_labels(frame)
        if labels is None:
            return 0
        x, y = position
        try:
            cell_idx = labels[y, x] - 1
        except IndexError:
            return 0
        if cell_idx < 0:
            return 0
        cells_info = self.get_cells_info(frame)
        if cells_info is not None:
            current_label = cells_info.label[cell_idx]
            cells_with_same_label_index = cells_info.query("label == %d and valid == 1 and empty_cell == 0" % new_label).index
            if len(cells_with_same_label_index) > 0:
                cells_info.at[cells_with_same_label_index[0],"label"] = current_label
            cells_info.at[cell_idx, "label"] = new_label
            # Updating cell label in subsequent frames
            for future_frame in range(frame + 1, self.number_of_frames + 1):
                cells_info = self.get_cells_info(future_frame)
                if cells_info is not None:
                    cell_idx = cells_info.query("label == %d and valid == 1 and empty_cell == 0" % current_label).index
                    if len(cell_idx) > 0:
                        cells_with_same_label_index = cells_info.query("label == %d and valid == 1 and empty_cell == 0" % new_label).index
                        if len(cells_with_same_label_index) > 0:
                            cells_info.at[cells_with_same_label_index[0], "label"] = current_label
                        cells_info.at[cell_idx[0], "label"] = new_label
                    else:
                        break
        return 0

    def fix_cell_id_in_events(self):
        if self.events is None:
            return 0
        for event_idx in self.events.index:
            start_frame = self.events.start_frame[event_idx]
            end_frame = self.events.end_frame[event_idx]
            start_pos = int(np.round(self.events.start_pos_x[event_idx])), int(np.round(self.events.start_pos_y[event_idx]))
            end_pos = int(np.round(self.events.end_pos_x[event_idx])), int(np.round(self.events.end_pos_y[event_idx]))
            daughter_pos = int(np.round(self.events.daughter_pos_x[event_idx])), int(np.round(self.events.daughter_pos_y[event_idx]))
            cell_id = self.get_cell_id_by_position(start_frame, start_pos)
            cell_end_id = self.get_cell_id_by_position(end_frame, end_pos)

            self.events.at[event_idx, "cell_id"] = cell_id
            if daughter_pos != (0,0):
                daughter_id = self.get_cell_id_by_position(end_frame, daughter_pos)
                if cell_id == daughter_id:
                    daughter_id = cell_end_id
                elif cell_id != cell_end_id:
                    self.fix_cell_label(end_frame, end_pos, cell_id)
                self.events.at[event_idx, "daughter_id"] = daughter_id
            else:
                if cell_end_id != cell_id:
                    self.fix_cell_label(end_frame, end_pos, cell_id)
        return 0

    def fix_cell_pos_in_events(self):
        if self.events is None:
            return 0
        for event_idx in self.events.index:
            start_frame = self.events.start_frame[event_idx]
            end_frame = self.events.end_frame[event_idx]
            cell_id = self.events.cell_id[event_idx]
            start_cell_data = self.get_cell_data_by_label(cell_id, start_frame)
            start_pos_x = start_cell_data.cx.values[0]
            start_pos_y = start_cell_data.cy.values[0]
            end_cell_data = self.get_cell_data_by_label(cell_id, end_frame)
            end_pos_x = end_cell_data.cx.values[0]
            end_pos_y = end_cell_data.cy.values[0]
            daughter_id = self.events.daughter_id[event_idx]
            self.events.at[event_idx, "start_pos_x"] = start_pos_x
            self.events.at[event_idx, "start_pos_y"] = start_pos_y
            self.events.at[event_idx, "end_pos_x"] = end_pos_x
            self.events.at[event_idx, "end_pos_y"] = end_pos_y
            if daughter_id > 0:
                daughter_data = self.get_cell_data_by_label(cell_id, end_frame)
                daughter_pos_x = daughter_data.cx.values[0]
                daughter_pos_y = daughter_data.cy.values[0]
                self.events.at[event_idx, "daughter_pos_x"] = daughter_pos_x
                self.events.at[event_idx, "daughter_pos_y"] = daughter_pos_y
        return 0

    def calc_cell_types(self, type_marker_image, frame_number, type_name, threshold=0.1,
                        percentage_above_threshold=90, peak_window_size=0):
        cells_info = self.get_cells_info(frame_number)
        labels = self.get_labels(frame_number)
        cell_types = self.get_cell_types(frame_number)
        if cell_types is None:
            cell_types = INVALID_TYPE_INDEX * np.ones_like(labels).astype(np.uint8)
        new_type = type_name not in self.type_names
        if new_type:
            self.type_names.append(type_name)
            type_index = len(self.type_names) - 1
        else:
            type_index = self.type_names.index(type_name)

        def percentile_intensity(regionmask, intensity):
            return np.percentile(intensity[regionmask], 100 - percentage_above_threshold)
        properties = regionprops_table(labels, intensity_image=type_marker_image,
                                       properties=('label', 'intensity_mean'),
                                       extra_properties=(percentile_intensity,))
        cell_indices = properties['label'] - 1
        if new_type:
            for cell_index in cell_indices:
                cells_info.at[cell_index, "mean_intensity"].append(properties['intensity_mean'])

        areas = cells_info.area.to_numpy()
        mean_area = np.mean(areas)
        max_area = self.max_cell_area * mean_area
        min_area = self.min_cell_area * mean_area
        old_valid = cells_info.valid.to_numpy() == 1
        new_valid = np.logical_and(areas < max_area, areas > min_area)
        updated_labels = cell_indices[np.logical_and(new_valid, ~old_valid)] + 1
        self.find_neighbors(frame_number, only_for_labels=updated_labels)
        cells_info.loc[:, "valid"] = new_valid.astype(int)
        max_brightness = np.percentile(type_marker_image, 99)
        threshold = threshold * max_brightness
        marker_intensities = properties["percentile_intensity"]
        pos_indices = cell_indices[marker_intensities > threshold]
        neg_indices = cell_indices[marker_intensities <= threshold]
        if peak_window_size > 0:
            local_maxima = find_local_maxima(type_marker_image, window_size=peak_window_size)
            indices_with_local_maximum = np.unique(labels[local_maxima]) - 1
            indices_with_local_maximum = indices_with_local_maximum[indices_with_local_maximum > 0]
            neg_indices = np.union1d(neg_indices, np.setdiff1d(pos_indices, indices_with_local_maximum))
            pos_indices = np.intersect1d(pos_indices, indices_with_local_maximum)

        # store inferred pos cells by bitwise or (changing the relevant digit in the binary representation to 1)
        current_type = self.cells_info.loc[pos_indices, "type"].to_numpy()
        new_type = change_type(current_type, type_index, is_positive=True)
        self.cells_info.loc[pos_indices, "type"] = new_type

        # store inferred neg cells by bitwise or (changing the relevant digit in the binary representation to 1)
        current_type = self.cells_info.loc[neg_indices, "type"].to_numpy()
        new_type = change_type(current_type, type_index, is_positive=False)
        self.cells_info.loc[neg_indices, "type"] = new_type
        self.update_cell_types_by_cells_info()
        return 0

    def update_cell_types_by_cells_info(self):
        valid_cells_info = self.cells_info.query("valid == 1")[["label", "type"]]
        for type_index in range(np.max(valid_cells_info.type) + 1):
            type_labels = valid_cells_info.query("type == %d" % type_index).to_numpy()
            self.cell_types[np.isin(self.labels, type_labels)] = type_index
        invalid_cells_labels = self.cells_info.query("valid == 0").label.to_numpy()
        self.cell_types[np.isin(self.labels, invalid_cells_labels)] = INVALID_TYPE_INDEX

    def fix_cell_types_after_tracking(self, window_size=11, consistency_threshold=0.5, min_frame_for_diff_detection=10, min_frames_to_change_type=3):
        KEEP_TYPE = -1
        DIFF_TYPE = -2

        # Get cell types from each frame:
        types_over_time = None
        number_of_cells = 0
        valid_frame_index = 0
        for frame in range(1, self.number_of_frames + 1):
            if not self.is_frame_valid(frame):
                continue
            cells_info = self.get_cells_info(frame)
            if cells_info is None:
                continue
            frame_type_data = cells_info.query("valid == 1 and empty_cell == 0").loc[:,["label", "type"]].copy()
            if types_over_time is None:
                number_of_cells = int(np.max(frame_type_data["label"]))
                types_over_time = KEEP_TYPE * np.ones((number_of_cells, self.get_number_of_valid_frames()))
            else:
                frame_number_of_cells = int(np.max(frame_type_data["label"]))
                if number_of_cells < frame_number_of_cells:
                    types_over_time = np.vstack([types_over_time,np.zeros((frame_number_of_cells - number_of_cells,
                                                                           self.get_number_of_valid_frames()))])
                    number_of_cells = frame_number_of_cells
            maximum_type_index = np.max(frame_type_data["type"])
            for type_index in range(maximum_type_index):
                type_labels = frame_type_data.query("type == %d" % type_index).label.to_numpy().astype(int)
                types_over_time[type_labels - 1, valid_frame_index] = type_index
            valid_frame_index += 1


        # Majority vote on moving window
        type_vote = [None]*maximum_type_index
        for type_index in range(maximum_type_index):
            type_vote[type_index] = convolve1d(types_over_time == type_index, np.ones((window_size,)), axis=1, mode='nearest')
        invalid_vote = convolve1d((types_over_time == -1).astype(int), np.ones((window_size,)), axis=1, mode='nearest')
        # first and last frames do not count because convolution enhances their "power"
        for type_index in range(maximum_type_index):
            type_vote[type_index][:,:window_size // 2] = 0
            type_vote[type_index][:,-window_size // 2:] = 0
        result = np.argmax(np.dstack([invalid_vote] + type_vote), axis=2)

        # Finding differentiation candidates - Frames where cell type switches from SC to HC
        diff_candidates = np.diff(result, axis=1, append=0) < 0

        # Scoring each candidate. score = %frames where cell was SC before candidate frame + %frames where cell was HC after candidate frame
        initial_types = result[:,0]
        final_types = result[:,-1]
        valid_frames_for_cell = np.sum((result >= 0).astype(int), axis=1)
        cumSC = np.cumsum((result == initial_types).astype(int),axis=1)
        cumHC = np.fliplr(np.cumsum(np.fliplr((result == final_types).astype(int)), axis=1))
        consistency_scores = np.zeros(diff_candidates.shape)
        consistency_scores[diff_candidates] = (cumSC[diff_candidates] + cumHC[diff_candidates])

        # Keeping only the best candidate from each cell and only if its consistency score is above threshold
        max_scores = np.max(consistency_scores, axis=1)/valid_frames_for_cell
        diff_frames_candidates = np.argmax(consistency_scores, axis=1)
        diff_cells_label = np.argwhere(np.logical_and(max_scores > consistency_threshold,
                                                      valid_frames_for_cell > min_frame_for_diff_detection)) + 1
        diff_frames = diff_frames_candidates[diff_cells_label - 1]

        # Deciding if non-differentiating cells are HCs, SCs or Unknown
        votes_for_each_type = np.apply_along_axis(np.bincount, axis=1, arr=result, minlength=3)
        max_votes = np.max(votes_for_each_type[:,1:], axis=1)
        new_types = np.argmax(votes_for_each_type[:,1:], axis=1)+1
        new_types[max_votes < min_frames_to_change_type] = KEEP_TYPE
        new_types[diff_cells_label - 1] = DIFF_TYPE
        diff_type_diff_frames = np.zeros(new_types.shape)
        diff_type_diff_frames[diff_cells_label - 1] = diff_frames

        # Update cell types
        valid_frame_index = 0
        for frame in range(1, self.number_of_frames + 1):
            if not self.is_frame_valid(frame):
                continue
            cells_info = self.get_cells_info(frame)
            if cells_info is None:
                continue
            labels_map = self.get_labels(frame)
            cell_types = self.get_cell_types(frame)
            if cell_types is None or labels is None:
                continue
            # update cell types in cells info
            valid_cells = cells_info.query("valid == 1 and empty_cell == 0")
            labels = valid_cells.label.to_numpy().astype(int)
            for type_index in range(maximum_type_index):
                change_to_type = np.zeros((valid_cells.shape[0],))
                change_to_type[new_types[labels - 1] == type_index] = 1
                change_to_diff = np.zeros((valid_cells.shape[0],))
                change_to_diff[new_types[labels - 1] == DIFF_TYPE] = 1
                before_diff_frame = np.zeros((valid_cells.shape[0],))
                before_diff_frame[diff_type_diff_frames[labels - 1] > valid_frame_index] = 1
                change_to_type[np.logical_and(np.logical_and(change_to_diff == 1, before_diff_frame == 1),
                                              type_index == initial_types)] = 1
                change_to_type[np.logical_and(np.logical_and(change_to_diff == 1, before_diff_frame == 0),
                                              type_index == final_types)] = 1
                valid_indices = valid_cells.index.to_numpy()
                type_indices = valid_indices[change_to_type == 1]
                self.cells_info.loc[type_indices, "type"] = type_index
                cell_types[np.isin(labels_map, type_indices + 1)] = type_index
            valid_frame_index += 1
        return 0

    def find_second_order_neighbors(self, frame, cells=None, cell_type='all', positive_for_type=True):
        cell_info = self.get_cells_info(frame)
        if cell_info is None:
            return None
        if cells is None:
            cells = cell_info.query("valid == 1")
        second_order_neighbors = [set() for i in range(cells.shape[0])]
        index = 0
        for i, row in cells.iterrows():
            for neighbor in list(row.neighbors):
                if cell_info.valid[neighbor - 1] == 0:
                    continue
                second_neighbors = np.array(list(cell_info.neighbors[neighbor - 1]))
                if cell_type == 'all':
                    valid_neighbors = second_neighbors[(cell_info.valid[second_neighbors - 1] == 1).to_numpy()]
                else:
                    type_index = self.type_name_to_index(cell_type)
                    second_neighbors_info = cell_info.iloc[second_neighbors - 1]
                    if positive_for_type:
                        valid_neighbors_info = second_neighbors_info.loc[
                            is_positive_for_type(second_neighbors_info.type.to_numpy(), type_index)].query("valid == 1")
                    else:
                        valid_neighbors_info = second_neighbors_info.loc[
                            ~is_positive_for_type(second_neighbors_info.type.to_numpy(), type_index)].query("valid == 1")
                    valid_neighbors = valid_neighbors_info.index.to_numpy() + 1
                second_order_neighbors[index] = second_order_neighbors[index].union(set(valid_neighbors))
            second_order_neighbors[index].difference(row.neighbors)
            if i + 1 in second_order_neighbors[index]:
                second_order_neighbors[index].remove(i + 1)
            index += 1
        return second_order_neighbors

    @staticmethod
    def find_nearest_neighbors_using_voroni_tesselation(cells):
        neighbors = [set() for i in range(cells.shape[0])]
        if cells.shape[0] < 4: # not enough points for Voroni tesselation
            return neighbors
        indices_in_original_table = cells.index.to_numpy()
        centers_x = cells.cx.to_numpy()
        centers_y = cells.cy.to_numpy()
        centers = np.hstack((centers_x.reshape((centers_x.size, 1)), centers_y.reshape(centers_y.size, 1)))
        neighbors = [set() for i in range(cells.shape[0])]
        voroni_diagram = Voronoi(centers)
        for neighbors_pair in voroni_diagram.ridge_points:
            first, second = neighbors_pair
            neighbors[first].add(indices_in_original_table[second] + 1)
            neighbors[second].add(indices_in_original_table[first] + 1)
        return neighbors


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

    def draw_cell_types(self, frame_number, type_name=""):
        type_index = self.type_name_to_index(type_name)
        if type_index < 0:
            type_index = 0
        cell_types = self.get_cell_types(frame_number)
        positive_image = (is_positive_for_type(cell_types, type_index)) * np.array(POS_COLOR).reshape((3, 1, 1))
        negative_image = (np.logical_and(~is_positive_for_type(cell_types, type_index),
                                         ~(cell_types == INVALID_TYPE_INDEX))) * np.array(NEG_COLOR).reshape((3, 1, 1))
        return positive_image + negative_image

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
        if cell_label == 0:
            return self.draw_all_cell_tracking(frame_number)
        labels = self.get_labels(frame_number)
        if labels is None:
            return 0
        img = np.zeros(labels.shape)
        if labels is None:
            return img
        cell = self.get_cell_data_by_label(cell_label, frame_number)
        if cell is None or cell.empty_cell.values[0] == 1:
            return img
        else:
            rr, cc = disk((cell.cy.values[0], cell.cx.values[0]), radius, shape=img.shape)
            img[rr, cc] = 1
        return img[np.newaxis, :,:] * np.array(TRACK_COLOR).reshape((3,1,1))

    def draw_all_cell_tracking(self, frame):
        track_labels = self.get_trackking_labels(frame)
        colors_num = len(TRACKING_COLOR_CYCLE)
        output = np.zeros((3,) + track_labels.shape)
        for i in range(colors_num):
            for j in range(3):
                output[j][track_labels%colors_num == i] = TRACKING_COLOR_CYCLE[i][j]
        # removing color from background
        for j in range(3):
            output[j][track_labels == 0] = 0
        return output

    def draw_marking_points(self, frame_number, radius=5):
        labels = self.get_labels(frame_number)
        img = np.zeros(labels.shape)
        if labels is None:
            return img
        for point in self.shape_fitting_points:
            rr, cc = disk((point[1], point[0]), radius, shape=img.shape)
            img[rr, cc] = 1
        return img[np.newaxis, :, :] * np.array(MARKING_COLOR).reshape((3, 1, 1))

    def add_segmentation_line(self, frame, point1, point2=None, initial=False, final=False, hc_marker_image=None,
                              hc_threshold=0.1, percentile_above_threshold=90):
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
                                                           hc_threshold, percentile_above_threshold)
            self._finished_last_line_addition = True
        return int(points_too_far)

    def remove_segmentation_line(self, frame, point1, part_of_undo=False):
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
            neighborhood = labels[max(0,y-1):min(y+2, labels.shape[0]), max(0,x-1):min(x+2, labels.shape[1])]
            unique_neighborhood = np.unique(neighborhood[neighborhood > 0])
            neighbors_relative_indices = np.array(np.where(neighborhood == 0))
            if y == 0:
                neighbors_relative_indices[0] += 1
            if x == 0:
                neighbors_relative_indices[1] += 1
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
                                                    part_of_undo)
        return 0

    def change_cell_type(self, frame, pos, type_name):
        labels = self.get_labels(frame)
        if labels is None:
            return 0
        x,y = pos
        cell_idx = labels[y, x] - 1
        if cell_idx < 0:
            return 0
        cells_info = self.get_cells_info(frame)
        cell_types = self.get_cell_types(frame)
        if cells_info is not None:
            type_index = self.type_name_to_index(type_name)
            try:
                current_type = cells_info.type[cell_idx]
                is_valid = cells_info.valid[cell_idx]
                if type_name == INVALID_TYPE_NAME:
                    new_type_for_map = INVALID_TYPE_INDEX
                    new_type = current_type
                else:
                    new_type = change_type(current_type, type_index, ~is_positive_for_type(current_type, type_index))
                    new_type_for_map = new_type
                self.cells_info.at[cell_idx, "type"] = new_type
                if cell_types is not None:
                    self.cell_types[labels == cell_idx + 1] = new_type_for_map
                if not is_valid:
                    self.cells_info.at[cell_idx, "valid"] = 1
                    self.find_neighbors(frame, only_for_labels=np.array([cell_idx + 1]))
            except IndexError:
                return 0
        return 0

    def make_invalid_cell(self, frame, pos):
        labels = self.get_labels(frame)
        if labels is None:
            return 0
        x, y = pos
        cell_idx = labels[y, x] - 1
        cells_info = self.get_cells_info(frame)
        if cells_info is not None:
            try:
                self.cells_info.at[cell_idx, "valid"] = 0
            except IndexError:
                return 0
        cell_types = self.get_cell_types(frame)
        if cell_types is not None:
            self.cell_types[labels == cell_idx + 1] = INVALID_TYPE_INDEX
        return 0

    def remove_cells_outside_of_sensory_region(self, frame):
        labels = self.get_labels(frame)
        if labels is None:
            return 0
        cells_info = self.get_cells_info(frame)
        if cells_info is not None:
            outside_cells_indices = self.detect_non_sensory_region_cells(cells_info)
            self.cells_info.loc[outside_cells_indices, "valid"] = 0
            cell_types = self.get_cell_types(frame)
            if cell_types is not None:
                self.cell_types[np.isin(labels,outside_cells_indices + 1)] = INVALID_TYPE_INDEX
        return 0

    def update_after_segmentation_line_removal(self, cell1_label, cell2_label, frame,
                                               part_of_undo=False):
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
                try:
                    if new_label > 0:
                        cell1_info = cell_info.iloc[cell1_label - 1]
                        cell2_info = cell_info.iloc[cell2_label - 1]
                        area1 = cell1_info.area
                        area2 = cell2_info.area
                        type1 = cell1_info.type
                        type2 = cell2_info.type
                        cell_info.at[new_label - 1, "area"] = area1 + area2
                        cell_info.at[new_label - 1, "perimeter"] = cell1_info.perimeter + cell2_info.perimeter - removed_line_length
                        cell_info.at[new_label - 1, "cx"] = (cell1_info.cx*area1 + cell2_info.cx*area2)/\
                                                            (area1 + area2)
                        cell_info.at[new_label - 1, "cy"] = (cell1_info.cy*area1 + cell2_info.cy*area2)/\
                                                            (area1 + area2)
                        mean_area = np.mean(cell_info.area.to_numpy())
                        max_area = self.max_cell_area * mean_area
                        min_area = self.min_cell_area * mean_area
                        valid = min_area < area1 + area2 < max_area
                        cell_info.at[new_label - 1, "valid"] = valid
                        new_type = max(type1, type2)
                        cell_info.at[new_label - 1, "type"] = new_type
                        if cell_types is not None:
                            cell_types[labels == new_label] = new_type if valid else INVALID_TYPE_INDEX
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
                except IndexError:
                    return 0
            else:
                new_label = cell1_label
                if part_of_undo:
                    labels[labels == -1] = new_label
                    labels[labels < 0] += 1
                    if not self._finished_last_line_addition:
                        self._finished_last_line_addition = True
                        return 0

        return 0

    def update_after_adding_segmentation_line(self, cell_label, frame):
        labels = self.get_labels(frame)
        cell_info = self.get_cells_info(frame)
        cell_types = self.get_cell_types(frame)
        if labels is not None:
            if cell_info is None:
                new_label = np.max(labels) + 1
                cell_indices = np.argwhere(labels == cell_label)
                bounding_box_min_row = np.min(cell_indices[:, 0]).astype(int)
                bounding_box_min_col = np.min(cell_indices[:, 1]).astype(int)
                bounding_box_max_row = np.max(cell_indices[:, 0]).astype(int) + 1
                bounding_box_max_col = np.max(cell_indices[:, 1]).astype(int) + 1
            else:
                empty_indices = np.argwhere(cell_info.empty_cell.to_numpy() == 1)
                if len(empty_indices) > 0:
                    new_label = empty_indices[0,0] + 1
                else:
                    new_label = cell_info.shape[0] + 1
                cell = cell_info.iloc[cell_label - 1]
                bounding_box_min_row = int(cell.bounding_box_min_row)
                bounding_box_min_col = int(cell.bounding_box_min_col)
                bounding_box_max_row = int(cell.bounding_box_max_row)
                bounding_box_max_col = int(cell.bounding_box_max_col)
            region_first_row = max(0,bounding_box_min_row-2)
            region_first_col = max(0,bounding_box_min_col-2)
            region_last_row = bounding_box_max_row+2
            region_last_col = bounding_box_max_col+2
            cell_region = labels[region_first_row:region_last_row, region_first_col:region_last_col]
            new_region_labels = label_image_regions((cell_region != 0).astype(int), connectivity=1, background=0)
            cell1_label = np.min(new_region_labels[cell_region == cell_label])
            cell2_label = np.max(new_region_labels[cell_region == cell_label])
            cell_region[new_region_labels == cell1_label] = cell_label
            cell_region[new_region_labels == cell2_label] = new_label
            labels[region_first_row:region_last_row, region_first_col:region_last_col] = cell_region
            if cell_info is not None:
                try:

                    properties = regionprops_table(cell_region, properties=("label", "area", "perimeter",
                                                                            "centroid", "bbox"))
                    mean_area = np.mean(cell_info.area.to_numpy())
                    max_area = self.max_cell_area * mean_area
                    min_area = self.min_cell_area * mean_area
                    for region_index, region_label in enumerate(properties["label"]):
                        if region_label == cell_label:
                            cell_info.at[cell_label - 1, "area"] = properties["area"][region_index]
                            cell_info.at[cell_label - 1, "perimeter"] = properties["perimeter"][region_index]
                            cell_info.at[cell_label - 1, "cx"] = properties["centroid-1"][region_index] + region_first_col
                            cell_info.at[cell_label - 1, "cy"] = properties["centroid-0"][region_index] + region_first_row
                            cell_info.at[cell_label - 1, "bounding_box_min_row"] = properties["bbox-0"][region_index] + region_first_row
                            cell_info.at[cell_label - 1, "bounding_box_min_col"] = properties["bbox-1"][region_index] + region_first_col
                            cell_info.at[cell_label - 1, "bounding_box_max_row"] = properties["bbox-2"][region_index] + region_first_row
                            cell_info.at[cell_label - 1, "bounding_box_max_col"] = properties["bbox-3"][region_index] + region_first_col
                            cell_valid = min_area < properties["area"][region_index] < max_area
                            cell_info.at[cell_label - 1, "valid"] = int(cell_valid)
                        elif region_label == new_label:
                            new_cell_valid = min_area < properties["area"][region_index] < max_area
                            new_cell_info = {"area": properties["area"][region_index],
                                             "label": new_label,
                                             "perimeter": properties["perimeter"][region_index],
                                             "cx": properties["centroid-1"][region_index] + region_first_col,
                                             "cy": properties["centroid-0"][region_index] + region_first_row,
                                             "bounding_box_min_row": properties["bbox-0"][region_index] + region_first_row,
                                             "bounding_box_min_col": properties["bbox-1"][region_index] + region_first_col,
                                             "bounding_box_max_row": properties["bbox-2"][region_index] + region_first_row,
                                             "bounding_box_max_col": properties["bbox-3"][region_index] + region_first_col,
                                             "valid": int(new_cell_valid),
                                             "empty_cell": 0,
                                             "neighbors": set(),
                                             "n_neighbors": 0,
                                             "type": int(cell.type)}
                            cell_info.loc[new_label - 1] = pd.Series(new_cell_info)
                    old_cell_neighbors = list(cell_info.neighbors[cell_label - 1].copy())
                    for neighbor_label in old_cell_neighbors:
                        cell_info.at[neighbor_label - 1, "neighbors"].remove(cell_label)
                    cell_info.at[cell_label - 1, "neighbors"] = set()
                    need_to_update_neighbors = list(cell_info.neighbors[cell_label]) + [cell_label, new_label]
                    self.find_neighbors(frame, labels_region=cell_region, only_for_labels=need_to_update_neighbors)
                    if cell_types is not None:
                        cell_types[labels == cell_label] = cell.type if cell_valid else INVALID_TYPE_INDEX
                        cell_types[labels == new_label] = cell.type if new_cell_valid else INVALID_TYPE_INDEX
                    return 0
                except IndexError:
                    return 0

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
        self.remove_segmentation_line(frame, line_pixel, part_of_undo=True)


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
            for i in [max(y-distance, 0), min(y+distance, labels.shape[0])]:
                for j in range(max(x-distance, 0), min(x+distance+1, labels.shape[1])):
                    if labels[i, j] == 0:
                        return j, i
            for j in [max(x-distance, 0), min(x+distance, labels.shape[1])]:
                for i in range(max(y-distance, 0), min(y+distance+1, labels.shape[0])):
                    if labels[i, j] == 0:
                        return j, i
        if distance_limit > 0:
            return None, None
        else:
            edges = [0, labels.shape[0], 0, labels.shape[1]]
            if nearest_edge < 2:
                return edges[nearest_edge], y
            else:
                return x, edges[nearest_edge]

    def get_shape_fitting_results(self, frame):
        return self.shape_fitting_results[frame - 1]

    def start_shape_fitting(self):
        self.shape_fitting_points = []
        self.shape_fitting_normalization = []

    def add_shape_fitting_point(self, frame, pos, marking_target):
        if marking_target == "Cells":
            cell = self.get_cell_by_pixel(pos[0], pos[1], frame)
            if cell.shape[0] > 0:
                centroid = (cell.cx, cell.cy)
            else:
                return 0
            area = cell.area
            self.shape_fitting_points.append(centroid)
            self.shape_fitting_normalization.append(area)
        else:
            self.shape_fitting_points.append(pos)
        return 0

    @staticmethod
    def calc_standard_error(der, cov_matrix):
        der_row_vector = der.reshape((1,der.size))
        der_column_vector = der.reshape((der.size, 1))
        return np.sqrt(der_row_vector.dot(cov_matrix.dot(der_column_vector)))[0,0]

    def end_shape_fitting(self, frame, shape, ax, name):
        X = np.array([pos[0] for pos in self.shape_fitting_points])
        Y = np.array([pos[1] for pos in self.shape_fitting_points])
        if self.shape_fitting_normalization:
            norm_factor = np.array(self.shape_fitting_normalization).mean()
        else:
            norm_factor = 1
        if shape == "ellipse":
            res = self.fit_an_ellipse(X, Y, ax, norm_factor)
        elif shape == "circle arc":
            res = self.fit_a_circle_arc(X, Y, ax, norm_factor)
        elif shape == "line":
            res = self.fit_a_line(X, Y, ax, norm_factor)
        elif shape == "spline":
            res = self.fit_a_spline(X, Y, ax, norm_factor)
        self.shape_fitting_results[frame - 1][name] = res
        return res

    def fit_a_line(self, X, Y, ax, norm_factor):

        horizontal = np.max(X) - np.min(X) > np.max(Y) - np.min(Y)

        # Fitting a line
        if horizontal:
            params, cov = np.polyfit(X, Y, 1, rcond=None, cov=True)
            slope = params[0]
            y_cross = params[1]
            x_cross = -params[1]/params[0]
            params_err =np.sqrt(np.diagonal(cov))
            slope_err = params_err[0]
            y_cross_err = params_err[1]
            x_cross_der = np.array([params[1]/params[0]**2, -1/params[0]])
            x_cross_err = np.sqrt(np.sum((params_err*x_cross_der)**2))
            chi_sqr = np.sum((Y - params[0]*X - params[1])**2)/(params[0]**2 + 1)
        else:
            params, cov = np.polyfit(Y, X, 1, rcond=None, cov=True)
            slope = 1/params[0]
            y_cross = -params[1]/params[0]
            x_cross = params[1]
            params_err = np.sqrt(np.diagonal(cov))
            slope_err = params_err[0]*slope**2
            x_cross_err = params_err[1]
            y_cross_der = np.array([params[1] / params[0] ** 2, -1 / params[0]])
            y_cross_err = np.sqrt(np.sum((params_err * y_cross_der)**2))
            chi_sqr = np.sum((X - params[0] * Y - params[1]) ** 2)/(params[0]**2 + 1)

        # Normalizing results
        chi_sqr /= (norm_factor * X.size)

        # plot the image
        ax.imshow(self.labels == 0)

        # Plot the  data
        ax.scatter(X, Y, label='Data Points')

        # Plot the least squares line

        if horizontal:
            x_coord = np.linspace(np.min(X), np.max(X), 300)
            y_coord = slope*x_coord + y_cross
        else:
            y_coord = np.linspace(np.min(Y), np.max(Y), 300)
            x_coord = y_coord/slope + x_cross

        ax.plot(x_coord, y_coord, 'r', linewidth=2, label="Fit")
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        res = {"slope": (slope, slope_err), "x cross": (x_cross, x_cross_err),
               "y cross": (y_cross, y_cross_err), "Chi square": (chi_sqr, 0), "N": (X.size, 0)}
        return res

    def fit_a_spline(self, X, Y, ax, norm_factor, cells_per_knot=10, max_iter=100):
        horizontal = np.max(X) - np.min(X) > np.max(Y) - np.min(Y)

        # Fitting a line
        if horizontal:
            params, cov = np.polyfit(X, Y, 1, rcond=None, cov=True)
            slope = params[0]
        else:
            params, cov = np.polyfit(Y, X, 1, rcond=None, cov=True)
            slope = 1 / params[0]

        # rotating points about their center of mass to make the spline orientation as horizontal as possible
        origin_x = np.mean(X)
        origin_y = np.mean(Y)
        angle = -np.arctan(slope)

        rot_x = origin_x + np.cos(angle) * (X - origin_x) - np.sin(angle) * (Y - origin_y)
        rot_y = origin_y + np.sin(angle) * (X - origin_x) + np.cos(angle) * (Y - origin_y)

        arg_sort = np.argsort(rot_x)

        # Sorting by x values since slipne fotting only works for increasing x values

        rot_sorted_x = rot_x[arg_sort]
        rot_sorted_y = rot_y[arg_sort]

        # fitting a spline
        knots = X.size//cells_per_knot + 2

        knots_in_spline = -1
        smoothing_param = X.size
        was_too_big = False
        was_too_small = False
        smoothing_param_factor = 2
        iter = 0
        while knots_in_spline != knots and iter < max_iter:
            spline = UnivariateSpline(rot_sorted_x, rot_sorted_y, s=smoothing_param)
            knots_in_spline = spline.get_knots().size
            if knots_in_spline < knots:
                was_too_small = True
                if was_too_big:
                    smoothing_param_factor -= (smoothing_param_factor - 1)/2
                    was_too_small = False
                smoothing_param /= smoothing_param_factor
            elif knots_in_spline > knots:
                was_too_big = True
                if was_too_small:
                    smoothing_param_factor -= (smoothing_param_factor - 1) / 2
                    was_too_big = False
                smoothing_param *= smoothing_param_factor
            iter += 1
        if iter >= max_iter:
            print("Warining: max iteration reached. Desired #knots: %d, actual #knots: %d" % (knots, knots_in_spline))


        chi_sqr = spline.get_residual()/(X.size*norm_factor)


        # plot the image
        ax.imshow(self.labels == 0)

        # Plot the  data
        ax.scatter(X, Y, label='Data Points')

        # Plot the least squares line

        x_coord_rot = np.linspace(np.min(rot_x), np.max(rot_x), 300)
        y_coord_rot = spline(x_coord_rot)
        x_coord = origin_x + np.cos(angle) * (x_coord_rot - origin_x) + np.sin(angle) * (y_coord_rot - origin_y)
        y_coord = origin_y - np.sin(angle) * (x_coord_rot - origin_x) + np.cos(angle) * (y_coord_rot - origin_y)

        ax.plot(x_coord, y_coord, 'r', linewidth=2, label="Fit")
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        res = {"Chi square": (chi_sqr, 0), "N": (X.size, 0), "knots":(knots_in_spline, 0)}

        return res

    def fit_a_circle_arc(self, X, Y, ax, norm_factor):
        # rescaling coordinates to avoid overflow
        rescale_factor = np.abs(max(np.max(X), np.max(Y)))
        rescaled_X = (X - np.mean(X)) / rescale_factor
        rescaled_Y = (Y - np.mean(Y)) / rescale_factor

        # Formulate and solve the least squares problem ||Ax - b ||^2
        A = np.column_stack([rescaled_X ** 2 + rescaled_Y ** 2, rescaled_X, rescaled_Y])
        b = np.ones_like(X)
        params = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()
        params_cov = np.linalg.inv(A.T.dot(A))
        fitting_points_max_distance_sqr = (np.max(rescaled_X) - np.min(rescaled_X)) ** 2 + \
                                          (np.max(rescaled_Y) - np.min(rescaled_Y)) ** 2
        linear_fit = params[0] * fitting_points_max_distance_sqr < 0.01

        # Obtaining canonical parameters
        curvature = 1/np.sqrt(1/params[0] + 0.25*(params[1]**2 + params[2]**2)/params[0]**2)
        slope = -params[1] / params[2]

        if linear_fit:
            chi_sqr = np.sum((params[1] * X + params[2] * Y - 1) ** 2)/(params[1]**2 + params[2]**2)
        else:
            chi_sqr = np.sum((np.sqrt((A.dot(params) - 1)/params[0] + 1/curvature**2) - 1/curvature)**2)


        # Obtaining canonical parameters errors
        curvature_der = -0.5 * curvature**3 * np.array([-1/params[0]**2 - 0.5*(params[1]**2 + params[2]**2)/params[0]**3,
                                                        0.5*params[1]/params[0]**2, 0.5*params[2]/params[0]**2])
        slope_der = np.array([0, -1/params[2], params[1]/(params[2]**2)])

        curvature_err = self.calc_standard_error(curvature_der, params_cov)
        slope_err = self.calc_standard_error(slope_der, params_cov)

        # Rescaling results
        curvature /= rescale_factor
        chi_sqr *= (rescale_factor**2)/(norm_factor * X.size)
        # plot the image
        ax.imshow(self.labels == 0)

        # Plot the  data
        ax.scatter(X, Y, label='Data Points')

        # Plot the least squares circle arc
        horizontal = np.max(rescaled_X) - np.min(rescaled_X) > np.max(rescaled_Y) - np.min(rescaled_Y)
        if horizontal:
            x_coord = np.linspace(np.min(rescaled_X), np.max(rescaled_X), 300)
            if linear_fit:
                y_coord = (1 - params[1] * x_coord) / params[2]
            else:
                y_coord_plus = (-params[2] + np.sqrt(
                    params[2] ** 2 - 4 * params[0] * (params[0] * x_coord ** 2 + params[1] * x_coord - 1))) \
                          / (2 * params[0])
                y_coord_minus = (-params[2] - np.sqrt(
                    params[2] ** 2 - 4 * params[0] * (params[0] * x_coord ** 2 + params[1] * x_coord - 1))) \
                               / (2 * params[0])
                y_coord = y_coord_plus if np.abs(np.min(rescaled_Y) - np.min(y_coord_plus)) < np.abs(np.min(rescaled_Y) - np.min(y_coord_minus)) else y_coord_minus
        else:
            y_coord = np.linspace(np.min(rescaled_Y), np.max(rescaled_Y), 300)
            if linear_fit:
                x_coord = (1 - params[2] * y_coord) / params[1]
            else:
                x_coord_plus = (-params[1] + np.sqrt(
                    params[1] ** 2 - 4 * params[0] * (params[0] * y_coord ** 2 + params[2] * y_coord - 1))) \
                          / (2 * params[0])
                x_coord_minus = (-params[1] - np.sqrt(
                    params[1] ** 2 - 4 * params[0] * (params[0] * y_coord ** 2 + params[2] * y_coord - 1))) \
                               / (2 * params[0])
                x_coord = x_coord_plus if np.abs(np.min(rescaled_X) - np.min(x_coord_plus)) < np.abs(
                    np.min(rescaled_X) - np.min(x_coord_minus)) else x_coord_minus

        x_coord = x_coord*rescale_factor + np.mean(X)
        y_coord = y_coord * rescale_factor + np.mean(Y)
        ax.plot(x_coord, y_coord, 'r', linewidth=2, label="Fit")
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        res = {"curvature": (curvature, curvature_err), "slope": (slope, slope_err),
               "Chi square": (chi_sqr, 0), "N": (X.size, 0)}
        return res


    def fit_an_ellipse(self, X, Y, ax, norm_factor): # taken from https://stackoverflow.com/questions/47873759/how-to-fit-a-2d-ellipse-to-given-points

        # rescaling coordinates to avoid overflow
        rescale_factor = np.abs(max(np.max(X), np.max(Y)))
        rescaled_X = (X - np.mean(X))/rescale_factor
        rescaled_Y = (Y - np.mean(Y))/rescale_factor

        # Least square fit
        A = np.column_stack([rescaled_X ** 2, rescaled_X * rescaled_Y, rescaled_Y ** 2, rescaled_X, rescaled_Y])
        b = np.ones_like(X)

        params, chi_sqr, _, _ = np.linalg.lstsq(A, b, rcond=None)
        params = params.squeeze()
        chi_sqr = chi_sqr[0]
        params_cov = np.linalg.inv(A.T.dot(A))

        # Obtaining canonical parameters
        a = params[0]*(params[4]**2) + params[2]*(params[3]**2) - params[1]*params[3]*params[4] - params[1]**2 + \
            4*params[0]*params[2]
        q = np.sqrt((params[0] - params[2]) ** 2 + params[1] ** 2)
        bplus = params[0] + params[2] + q
        bminus = params[0] + params[2] - q
        c = params[1]**2 - 4*params[0]*params[2]
        sqrt_2abplus = np.sqrt(2 * a * bplus)
        sqrt_2abminus = np.sqrt(2 * a * bminus)
        semi_major = -sqrt_2abplus/c
        semi_minor = -sqrt_2abminus/c
        center_x = (2*params[2]*params[3] - params[1]*params[4])/c
        center_y = (2*params[0]*params[4] - params[1]*params[3])/c
        phi = (params[2] - params[0] - q)/params[1]
        rotating_angle = np.arctan(phi) if params[1] != 0 else 0 if params[0] < params[2] else np.pi/2
        eccentricity = 2*(semi_major/semi_minor - 1)/3

        # Obtaining canonical parameters errors
        ader = np.array([params[4]**2 + 4*params[2], -params[3]*params[4] - 2*params[1], params[3]**2 + 4*params[0],
                         2*params[2]*params[3] - params[1]*params[4], 2*params[0]*params[4] - params[1]*params[3]])
        bplusder = np.array([1 + (params[0] - params[2])/q, params[1]/q, 1 - (params[0] - params[2])/q, 0, 0])
        bminusder = np.array([1 - (params[0] - params[2]) / q, -params[1] / q, 1 + (params[0] - params[2]) / q, 0, 0])
        cder = np.array([-4*params[2], 2*params[1], -4*params[0], 0, 0])
        phider = np.array([(-1 - (params[0] - params[2])/q)/params[1], -phi/params[1]-1/q,
                           (1 + (params[0] - params[2])/q)/params[1], 0, 0])
        semi_minor_der = (sqrt_2abminus/(c**2))*cder - 2*(bminus*ader + a*bminusder)/(sqrt_2abminus*c)
        semi_major_der = (sqrt_2abplus/(c**2))*cder - 2*(bplus*ader + a*bplusder)/(sqrt_2abplus*c)
        center_x_der = np.array([0, -params[4], 2*params[3], 2*params[2], -params[1]])/c - (center_x/c)*cder
        center_y_der = np.array([2*params[4], -params[3], 0,  - params[1], 2*params[0]])/c - (center_y/c)*cder
        rotating_angle_der = (1/(1 + phi**2))*phider
        eccentricity_der = 2*(semi_major_der/semi_minor - (semi_minor_der*semi_major)/semi_minor**2)/3

        semi_major_err = self.calc_standard_error(semi_major_der, params_cov)
        semi_minor_err = self.calc_standard_error(semi_minor_der, params_cov)
        center_x_err = self.calc_standard_error(center_x_der, params_cov)
        center_y_err = self.calc_standard_error(center_y_der, params_cov)
        rotating_angle_err = self.calc_standard_error(rotating_angle_der, params_cov)
        eccentricity_err = self.calc_standard_error(eccentricity_der, params_cov)

        #rescale back
        center_x = center_x * rescale_factor + np.mean(X)
        center_y = center_y * rescale_factor + np.mean(Y)
        center_x_err *= rescale_factor
        center_y_err *= rescale_factor
        semi_major *= rescale_factor
        semi_major_err *= rescale_factor
        semi_minor *= rescale_factor
        semi_minor_err *= rescale_factor
        chi_sqr *= rescale_factor**2 / (norm_factor * X.size)

        # plot the image
        ax.imshow(self.labels == 0)

        # Plot the data
        ax.scatter(X, Y, label='Data Points')
        # Plot the least squares ellipse
        angle = np.linspace(0, 2*np.pi, 300)
        x_coord = semi_major*np.cos(angle)*np.cos(rotating_angle) -\
                  semi_minor*np.sin(angle)*np.sin(rotating_angle) + center_x
        y_coord = semi_major*np.cos(angle)*np.sin(rotating_angle) +\
                  semi_minor*np.sin(angle)*np.cos(rotating_angle) + center_y

        ax.plot(x_coord, y_coord, 'r', linewidth=2, label="Fit")
        ax.plot([-semi_major*np.cos(rotating_angle) + center_x, semi_major*np.cos(rotating_angle) + center_x],
                [-semi_major*np.sin(rotating_angle) + center_y, semi_major*np.sin(rotating_angle) + center_y], 'm', linewidth=2, label="Major axis")
        ax.plot([-semi_minor*np.sin(rotating_angle) + center_x, semi_minor*np.sin(rotating_angle) + center_x],
                [semi_minor*np.cos(rotating_angle) + center_y, -semi_minor*np.cos(rotating_angle) + center_y], 'orange', linewidth=2, label="Minor axis")
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        res = {"semi-major": (semi_major, semi_major_err), "semi-minor": (semi_minor, semi_minor_err),
               "rotation angle": (-rotating_angle, rotating_angle_err),
               "center x": (center_x, center_x_err), "center y": (center_y, center_y_err),
               "eccentricity": (eccentricity, eccentricity_err), "Chi square": (chi_sqr, 0), "N": (X.size, 0)}
        return res

    def initialize_working_space(self):
        working_dir = get_temp_directory(self.data_path)
        os.mkdir(working_dir)
        return working_dir

    def load_labels_from_external_file(self, frame, origin_path):
        if os.path.isfile(origin_path):
            image, axes, image_shape, metadata = read_tiff(origin_path)
            labels = label_image_regions(image, background=255, connectivity=1)
            self.set_labels(frame, labels, reset_data=True)
        return self.labels

    def load_labels(self, frame_number):
        if self.data_in_memory:
            self.labels = self.labels_list[frame_number - 1]
        else:
            file_path = os.path.join(self.working_dir, "frame_%d_labels.npy" % frame_number)
            if os.path.isfile(file_path):
                self.labels = np.load(file_path)
            else:
                self.labels = None
        self.labels_frame = frame_number
        return self.labels

    def load_cell_types(self, frame_number):
        if self.data_in_memory:
            self.cell_types = self.cells_type_list[frame_number - 1]
        else:
            file_path = os.path.join(self.working_dir, "frame_%d_types.npy" % frame_number)
            if os.path.isfile(file_path):
                self.cell_types = np.load(file_path)
            else:
                self.cell_types = None
        self.cell_types_frame = frame_number
        if self.is_cell_types_from_old_version():
            self.update_cell_types_to_multitype_version()
        return self.cell_types

    def load_cells_info(self, frame_number, type_name=""):
        if self.data_in_memory:
            self.cells_info = self.cell_info_list[frame_number - 1]
        else:
            file_path = os.path.join(self.working_dir, "frame_%d_data.pkl" % frame_number)
            if os.path.isfile(file_path):
                self.cells_info = pd.read_pickle(file_path)
            else:
                self.cells_info = None
        self.cells_info_frame = frame_number
        if self.is_cell_info_from_old_version():
            self.update_cell_info_to_multitype_version(type_name)
            if type_name and len(self.type_names) == 0:
                self.type_names = [type_name]
        return self.cells_info

    def load_events(self):
        file_path = os.path.join(self.working_dir, "events_data.pkl")
        if os.path.isfile(file_path):
            self.events = pd.concat([self.events, pd.read_pickle(file_path)])
            self.events['source'] = self.events['source'].fillna('manual')
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

    def load_shape_fitting(self):
        file_path = os.path.join(self.working_dir, "shape_fitting_data.json")
        if os.path.isfile(file_path):
            with open(file_path) as file:
                self.shape_fitting_results = json.load(file)
        return self.shape_fitting_results

    def load_stage_loactions(self):
        file_name = "stage_locations_%s.pkl" % os.path.basename(self.data_path).replace(".tif", "")
        directory = os.path.dirname(self.data_path)
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            return pd.DataFrame(pd.read_pickle(file_path))
        else:
            return None

    def load_height_map(self):
        file_name = "zmap_%s.npy" % os.path.basename(self.data_path).replace(".tif", "")
        directory = os.path.dirname(self.data_path)
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            return np.load(file_path, mmap_mode="r")
        else:
            return None

    def remove_labels(self):
        if self.data_in_memory:
            self.labels_list[self.labels_frame - 1] = None
        else:
            file_path = os.path.join(self.working_dir, "frame_%d_labels.npy" % self.labels_frame)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except OSError as e:
                print(str(e))
                sleep(1)
                self.remove_labels()

    def remove_cell_types(self):
        if self.data_in_memory:
            self.cells_type_list[self.cell_types_frame - 1] = None
        else:
            file_path = os.path.join(self.working_dir, "frame_%d_types.npy" % self.cell_types_frame)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except OSError as e:
                print(str(e))
                sleep(1)
                self.remove_cell_types()

    def remove_cells_info(self):
        if self.data_in_memory:
            self.cell_info_list[self.cells_info_frame - 1] = None
        else:
            file_path = os.path.join(self.working_dir, "frame_%d_data.pkl" % self.cells_info_frame)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except OSError as e:
                print(str(e))
                sleep(1)
                self.remove_cells_info()

    def save_labels(self):
        if self.data_in_memory:
            self.labels_list[self.labels_frame - 1] = self.labels
        else:
            if self.labels is not None:
                file_path = os.path.join(self.working_dir, "frame_%d_labels.npy" % self.labels_frame)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    np.save(file_path, self.labels)
                except OSError as e:
                    print(str(e))
                    sleep(1)
                    self.save_labels()
        return 0

    def save_drifts(self):
        if self.drifts is not None:
            file_path = os.path.join(self.working_dir, "drifts.npy")
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                np.save(file_path, self.drifts)
            except OSError as e:
                print(str(e))
                sleep(1)
                self.save_drifts()
        return 0

    def save_valid_frames(self):
        if self.valid_frames is not None:
            file_path = os.path.join(self.working_dir, "valid_frames.npy")
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                np.save(file_path, self.valid_frames)
            except OSError as e:
                print(str(e))
                sleep(1)
                self.save_valid_frames()
        return 0

    def save_cell_types(self):
        if self.data_in_memory:
            self.cells_type_list[self.cell_types_frame - 1] = self.cell_types
        else:
            if self.cell_types is not None:
                file_path = os.path.join(self.working_dir, "frame_%d_types.npy" % self.cell_types_frame)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    np.save(file_path, self.cell_types)
                except OSError as e:
                    print(str(e))
                    sleep(1)
                    self.save_cell_types()
        return 0

    def save_cells_info(self):
        if self.data_in_memory:
            self.cell_info_list[self.cells_info_frame - 1] = self.cells_info
        else:
            if self.cells_info is not None:
                file_path = os.path.join(self.working_dir, "frame_%d_data.pkl" % self.cells_info_frame)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    self.cells_info.to_pickle(file_path)
                except OSError as e:
                    print(str(e))
                    sleep(1)
                    self.save_cells_info()
        return 0

    def save_events(self):
        file_path = os.path.join(self.working_dir, "events_data.pkl")
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            self.events.to_pickle(file_path)
        except OSError as e:
            print(str(e))
            sleep(1)
            self.save_events()
        return 0

    def save_shape_fitting(self):
        file_path = os.path.join(self.working_dir, "shape_fitting_data.json")
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            with open(file_path, "w") as file:
                json.dump(self.shape_fitting_results, file)
        except OSError as e:
            print(str(e))
            sleep(1)
            self.save_shape_fitting()

    def save(self, path):
        self.save_labels()
        self.save_cells_info()
        self.save_cell_types()
        self.save_events()
        self.save_drifts()
        self.save_valid_frames()
        self.save_shape_fitting()
        self.save_cell_type_names()
        if self.data_in_memory:
            self.save_data_from_memory()
        for percent_done in pack_archive_with_progress(self.working_dir, path.replace(".seg", "") + ".seg"):
            yield percent_done
        return 0

    def load(self, path, type_name=""):
        old_working_dir = self.working_dir
        self.working_dir = self.initialize_working_space()
        for percent_done in unpack_archive_with_progress(path, self.working_dir):
            yield percent_done
        old_files_list = os.listdir(old_working_dir)
        for file in old_files_list:
            if not os.path.exists(os.path.join(self.working_dir, file)):
                os.rename(os.path.join(old_working_dir, file), os.path.join(self.working_dir, file))
        rmtree(old_working_dir)
        if self.data_in_memory:
            self.load_data_to_memory()
        if self.labels_frame > 0:
            self.load_labels(self.labels_frame)
        if self.cell_types_frame > 0:
            self.load_cell_types(self.cell_types_frame)
        if self.cells_info_frame > 0:
            self.load_cells_info(self.cells_info_frame, type_name=type_name)
        self.load_events()
        self.load_drifts()
        self.load_valid_frames()
        self.load_shape_fitting()
        self.load_cell_type_names()
        return 0

    def save_data_from_memory(self):
        for index, labels in enumerate(self.labels_list):
            frame = index + 1
            if labels is None:
                continue
            file_path = os.path.join(self.working_dir, "frame_%d_labels.npy" % frame)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                np.save(file_path, labels)
            except OSError as e:
                print(str(e))
                sleep(1)
                np.save(file_path, labels)
        for index, cell_types in enumerate(self.cells_type_list):
            frame = index + 1
            if cell_types is None:
                continue
            file_path = os.path.join(self.working_dir, "frame_%d_types.npy" % frame)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                np.save(file_path, cell_types)
            except OSError as e:
                print(str(e))
                sleep(1)
                np.save(file_path, cell_types)
        for index, cell_info in enumerate(self.cell_info_list):
            frame = index + 1
            if cell_info is None:
                continue
            file_path = os.path.join(self.working_dir, "frame_%d_data.pkl" % frame)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                cell_info.to_pickle(file_path)
            except OSError as e:
                print(str(e))
                sleep(1)
                cell_info.to_pickle(file_path)

    def load_data_to_memory(self):
        for index in range(len(self.labels_list)):
            frame = index + 1
            file_path = os.path.join(self.working_dir, "frame_%d_labels.npy" % frame)
            if os.path.isfile(file_path):
                self.labels_list[index] = np.load(file_path)
            else:
                self.labels_list[index] = None
        for index in range(len(self.cells_type_list)):
            frame = index + 1
            file_path = os.path.join(self.working_dir, "frame_%d_types.npy" % frame)
            if os.path.isfile(file_path):
                self.cells_type_list[index] = np.load(file_path)
            else:
                self.cells_type_list[index] = None
        for index in range(len(self.cell_info_list)):
            frame = index + 1
            file_path = os.path.join(self.working_dir, "frame_%d_data.pkl" % frame)
            if os.path.isfile(file_path):
                self.cell_info_list[index] = pd.read_pickle(file_path)
            else:
                self.cell_info_list[index] = None
        return 0


    def flip_all_data(self):
        for frame in range(1, self.number_of_frames + 1):
            self.flip_frame_data(frame)
        self.drifts[:, [0, 1]] = self.drifts[:, [1, 0]]
        self.events.loc[:, ["start_pos_y", "start_pos_x", "end_pos_y", "end_pos_x",
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
            self.cells_info.loc[:,["cy", "cx", "bounding_box_min_col", "bounding_box_min_row",
                                  "bounding_box_max_col", "bounding_box_max_row"]] = self.cells_info.loc[:,
                                 ["cx", "cy", "bounding_box_min_row", "bounding_box_min_col",
                                  "bounding_box_max_row", "bounding_box_max_col"]].values
        else:
            print("cell info on frame %d are none" % frame)
        self.fix_cell_pos_in_events()
        self.save_events()
        self.save_labels()
        self.save_cell_types()
        self.save_cells_info()

    def fix_types_in_cell_info(self):
        for frame in range(1, self.number_of_frames + 1):
            cells_info = self.get_cells_info(frame)
            valid_cells = cells_info.query("valid == 1 and empty_cell == 0")
            cell_types = self.get_cell_types(frame)
            cell_centroids = valid_cells.loc[:, ["cy", "cx"]].to_numpy()
            types = cell_types[np.round(cell_centroids[:, 0]).astype(int),
                               np.round(cell_centroids[:, 1]).astype(int)]
            valid_cells_indices = valid_cells.index.to_numpy()
            for type_index in range(np.max(cell_types) + 1):
                indices = valid_cells_indices[types == type_index]
                self.cells_info.loc[indices, "type"] = type_index
            indices = valid_cells_indices[types == INVALID_TYPE_INDEX]
            self.cells_info.loc[indices, "valid"] = 0

            print("finished frame %d" % frame, flush=True)
        return 0

    def calculate_total_area_in_movie(self):
        area = 0
        for frame in range(1, self.number_of_frames + 1):
            if self.is_frame_valid(frame):
                cells_info = self.get_cells_info(frame)
                valid_cells = cells_info.query("valid == 1 and empty_cell == 0")
                area += np.sum(valid_cells.area.to_numpy())
        print("Total area = %f pixels^2" % area)
        return 0

    def save_event_statistics_data(self, ref_frames, output_dir):
        event_types = ["division", "delamination", "differentiation", "overall reference SC", "overall reference HC"]
        event_labels = ["division", "delamination", "differentiation", "reference_SC", "reference_HC"]
        features = [("area", "roundness"), ("HC contact length", "SC contact length"),
                    ("HC density", "HC type_fraction"), ("HC neighbors", "SC neighbors"),
                    ("n_neighbors",), ("HC second neighbors", "SC second neighbors"),
                    ("timing histogram",)]
        feature_labels = ["area_and_roundness", "contact_length_by_type", "HC_density_and_fraction",
                          "neighbors_by_type", "number_of_neighbors", "second_neighbors_by_type", "timing"]
        for event_type, event_label in zip(event_types, event_labels):
            for feature, feature_label in zip(features, feature_labels):

                x_feature = feature[0]
                if len(feature) > 1:
                    y_feature = feature[1]
                else:
                    y_feature = None
                x_radius = 200
                y_radius = 200
                if "reference" in event_type:
                    if x_feature == "timing histogram":
                        continue
                    for frame in ref_frames:
                        fig, ax = plt.subplots()
                        filename = "%s_%s_frame%d.png" % (feature_label, event_label, frame)
                        data_filename = "%s_%s_frame%d_data" % (feature_label, event_label, frame)
                        cell_type = "SC" if "SC" in event_type else "HC"
                        res, msg = self.plot_overall_statistics(frame, x_feature, y_feature, ax, intensity_img=None,
                                x_cells_type=cell_type, y_cells_type=cell_type, x_radius=x_radius, y_radius=y_radius)

                        fig.savefig(os.path.join(output_dir, filename))
                        if isinstance(res, pd.DataFrame):
                            res.to_pickle(os.path.join(output_dir, data_filename))
                        elif isinstance(res, np.ndarray):
                            np.save(os.path.join(output_dir, data_filename), res)
                else:
                    fig, ax = plt.subplots()
                    filename = "%s_%s.png" % (feature_label, event_label)
                    data_filename = "%s_%s_data" % (feature_label, event_label)
                    res, msg = self.plot_event_statistics(event_type, x_feature, x_radius, y_feature, y_radius, ax,
                                               intensity_images=None)

                    fig.savefig(os.path.join(output_dir, filename))
                    plt.close(fig)
                    if isinstance(res, pd.DataFrame):
                        res.to_pickle(os.path.join(output_dir, data_filename))
                    elif isinstance(res, np.ndarray):
                        np.save(os.path.join(output_dir, data_filename), res)

    def get_trackking_labels(self, frame):
        labels = self.get_labels(frame)
        cells_info = self.get_cells_info(frame)
        if labels is None or cells_info is None:
            return None
        cell_ids = np.insert(cells_info.label.to_numpy(), 0, 0)
        cells_id_in_order = cell_ids[np.searchsorted(cells_info.index.to_numpy() + 1, labels, side='right')]
        return cells_id_in_order

    def export_segmentation_to_matlab(self, outfolder, filename):
        out = dict()
        for frame in range(1, self.number_of_frames + 1):
            labels = self.get_trackking_labels(frame)
            out["frame%d" % frame] = labels.astype("uint16")
        out["valid_frames"] = self.valid_frames
        out["number_of_frames"] = self.number_of_frames
        savemat(os.path.join(outfolder, filename + '.mat'), out)
        return 0

    def export_segmentation_and_cell_types_to_tiff(self, outfolder, filename):
        out = np.zeros((self.number_of_frames, 2, 1, self.labels.shape[0], self.labels.shape[1]),dtype="uint16")
        for frame in range(1, self.number_of_frames + 1):
            if self.is_frame_valid(frame):
                labels = self.get_trackking_labels(frame)
                out[frame - 1, 0, 0, :,:] = labels.astype("uint16")
                cell_types = self.get_cell_types(frame)
                out[frame - 1, 1, 0, :, :] = cell_types.astype("uint16")
        save_tiff(os.path.join(outfolder, filename + '.tif'),out , axes="TCZYX", data_type="uint16")
        return 0

    def export_segmentation_to_tiff(self, outfolder, filename):
        labels = self.get_labels(1)
        out = np.zeros((self.number_of_frames, 1, 1, self.labels.shape[0], self.labels.shape[1]), dtype="uint16")
        for frame in range(1, self.number_of_frames + 1):
            if self.is_frame_valid(frame):
                labels = self.get_labels(frame)
                out[frame - 1, 0, 0, :, :] = labels.astype("uint16").T
        save_tiff(os.path.join(outfolder, filename + '.tif'), out, axes="TCZYX", data_type="uint16")
        return 0

    def export_segmentation_to_npy(self, outfolder, filename):
        out = [None] * self.number_of_frames
        for frame in range(1, self.number_of_frames + 1):
            labels = self.get_trackking_labels(frame)
            out[frame - 1] = labels.astype("uint16")
        out = np.array(out).astype("uint16")
        np.save(os.path.join(outfolder, filename + '.tif'), out)
        return 0

    def save_cell_type_names(self):
        if self.type_names is not None:
            file_path = os.path.join(self.working_dir, "cell_type_names.pkl")
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                with open(file_path, 'wb') as fp:
                    pickle.dump(self.type_names, fp)
            except OSError as e:
                print(str(e))
                sleep(1)
                self.save_cell_type_names()
        return 0

    def load_cell_type_names(self):
        file_path = os.path.join(self.working_dir, "cell_type_names.pkl")
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as fp:
                self.type_names = pickle.load(fp)
        return self.type_names

    def is_cell_info_from_old_version(self):
        if self.cells_info is None:
            return False
        return type(self.cells_info.at[0,"type"]) == str

    def is_cell_types_from_old_version(self):
        if self.cell_types is None:
            return False
        return np.max(self.cell_types) <=2 and np.min(self.cell_types) >=0

    def update_cell_info_to_multitype_version(self, type_name):
        self.cells_info.replace({"HC": 1, "SC": 0, "invalid": 0}, inplace=True)
        return 0

    def update_cell_types_to_multitype_version(self):
        self.cell_types[self.cell_types == 0] = INVALID_TYPE_INDEX
        self.cell_types[self.cell_types == 2] = 0
        return 0