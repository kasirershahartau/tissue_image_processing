# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 12:53:47 2021

@author: Shahar Kasirer, Anastasia Pergament 

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
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.interpolate import UnivariateSpline
from scipy.spatial import Voronoi
from skimage.draw import line, disk
from skimage.measure import regionprops_table, regionprops
from skimage.measure import label as label_image_regions
from skimage.registration import phase_cross_correlation
import zipfile
from basic_image_manipulations import read_tiff
from time import sleep
import json

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
                    "daughter_id": 0,
                    "source": "manual"}


TRACK_COLOR = (0, 1, 0)
NEIGHBORS_COLOR = (1, 1, 1)
HC_COLOR = (1, 0, 1)
SC_COLOR = (1, 1, 0)
MARKING_COLOR = (0.5, 0.5, 0.5)
EVENT_TYPES = ["ablation","division", "delamination", "differentiation", "delete event"]
EVENTS_COLOR = {"ablation": (1,1,0), "division": (0,0,1), "delamination": (1,0,0), "differentiation": (0,1,1)}
PIXEL_LENGTH = 0.1  # in microns. TODO: get from image metadata

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
    SPECIAL_FEATURES = ["shape index", "roundness", "neighbors from the same type", "HC neighbors", "SC neighbors", "contact length",
                        "HC contact length", "SC contact length", "Mean atoh intensity"]
    SPECIAL_X_ONLY_FEATURES = ["psi6"]
    SPECIAL_Y_ONLY_FEATURES = ["histogram"]
    GLOBAL_FEATURES = ["density", "type_fraction", "total_area"]
    CELL_TYPES = ["all", "HC", "SC"]
    FITTING_SHAPES = ["ellipse", "circle arc", "line", "spline"]

    def __init__(self, number_of_frames, data_path, max_cell_area=10, min_cell_area=0.1):
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

    def reset_all_data(self):
        self.cells_info = None
        self.labels = None
        self.cell_types = None
        self.cells_info_frame = 0
        self.labels_frame = 0
        self.cell_types_frame = 0
        old_working_dir = self.working_dir
        self.working_dir = self.initialize_working_space()
        rmtree(old_working_dir)
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
        try:
            cell_id = cells_info.label[cell_idx]
            return cell_id
        except IndexError:
            return 0


    def get_cell_centroid_by_id(self, frame, id):
        cells_info = self.get_cells_info(frame)
        cell = cells_info.query("label == %d" % id)
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

    def find_events_iterator(self, initial_frame=1, final_frame=-1):
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
        HC_id_previous_frame = valid_cells_last_frame.query("type == \"HC\"").label
        labels_previous_frame = np.copy(labels)
        edge_cells_id_previous_frame = cells_info.label[self.detect_edge_cells(labels)]
        for frame in range(initial_frame + 1, final_frame + 1):
            if not self.valid_frames[frame - 1]:
                continue
            adjacent_valid_frames = self.find_valid_frames(frame - 5, frame + 5)
            start_frame = np.min(adjacent_valid_frames)
            end_frame = np.max(adjacent_valid_frames)
            labels = self.get_labels(frame)
            cells_info = self.get_cells_info(frame)
            valid_cells_current_frame = cells_info.query("valid == 1 and empty_cell == 0")
            id_current_frame = valid_cells_current_frame.label
            edge_cells_id_current_frame = cells_info.label[self.detect_edge_cells(labels)]
            HC_id_current_frame = valid_cells_last_frame.query("type == \"HC\"").label

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
            yield frame
        return 0



    def calculate_frame_cellinfo(self, frame_number, hc_marker_image=None, hc_threshold=0.01, use_existing_types=False,
                                 percentage_above_HC_threshold=90):
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
            def percentile_intensity(regionmask, intensity):
                return np.percentile(intensity[regionmask], 100-percentage_above_HC_threshold)
            properties = regionprops_table(labels, intensity_image=hc_marker_image,
                                           properties=('label', 'area', 'perimeter', 'centroid', 'bbox'),
                                           extra_properties=(percentile_intensity,))
        else:
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
        self.find_neighbors(frame_number)
        if hc_marker_image is not None:
            self.calc_cell_types(hc_marker_image, frame_number, properties, hc_threshold, percentage_above_HC_threshold)

    def get_cell_data_by_label(self, cell_id, frame):
        cells_info = self.get_cells_info(frame)
        if cells_info is None:
            return None
        cells_with_matching_label = cells_info.query("label == %d" % cell_id)
        if cells_with_matching_label.shape[0] > 0:
            return cells_with_matching_label
        else:
            return None

    def plot_single_cell_data(self, cell_id, feature, ax, intensity_images=None):
        frames = np.arange(1, self.number_of_frames + 1)
        t = (frames - 1)*15
        data = self.get_single_cell_data(cell_id, frames, feature, intensity_images)
        t = t[~np.isnan(data)]
        data = data[~np.isnan(data)]
        ax.plot(t, data, '*')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel(feature)
        ax.set_title("%s of cell number %d" % (feature, cell_id))
        return pd.DataFrame({"Time": t, feature: data})

    def get_single_cell_data(self, cell_id, frames, feature, intensity_images=None):
        current_cell_info_frame = self.cells_info_frame
        data = np.zeros((len(frames),))
        for index,frame in enumerate(frames):
            cell = self.get_cell_data_by_label(cell_id, frame)
            if self.is_frame_valid(frame) and cell is not None:
                if intensity_images is None:
                    intensity_image = None
                else:
                    intensity_image = intensity_images[index]
                frame_data, msg = self.get_frame_data(frame, feature, cell, self.SPECIAL_FEATURES,
                                                      intensity_img=intensity_image)
                if msg:
                    return None, msg
                data[index] = frame_data[0]
            else:
                data[index] = np.nan
        self.get_cells_info(current_cell_info_frame)
        return data

    def plot_event_related_data(self, cell_id, event_frame, feature, frames_around_event, ax, intensity_images=None):
        event_data = self.events.query("cell_id == %d and start_frame <= %d and end_frame >= %d" %(cell_id, event_frame, event_frame))
        if event_data.shape[0] < 1:
            return None
        else:
            frames = np.arange(max(event_frame - frames_around_event, 0),
                               min(event_frame + frames_around_event + 1, self.number_of_frames + 1))
            t = (frames - 1) * 15
            data = self.get_single_cell_data(cell_id, frames, feature, intensity_images)
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
                daughter_data = self.get_single_cell_data(daughter_id, daughter_frames, feature)
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

    def find_event_frame(self, event):
        start_frame = event.start_frame.values[0]
        end_frame = event.end_frame.values[0]
        event_type = event.type.values[0]
        for frame in range(start_frame, end_frame+1):

            if event_type == "delamination":
                pass

    def get_frame_data(self, frame, feature, valid_cells, special_features, global_features=[],
                      for_histogram=False, reference=None, intensity_img=None):
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
                data = self.calculate_n_neighbors_from_type(frame, valid_cells, cell_type='HC')
            elif feature == "SC neighbors":
                data = self.calculate_n_neighbors_from_type(frame, valid_cells, cell_type='SC')
            elif feature == "Mean atoh intensity":
                data = self.calculate_mean_intensity(frame, valid_cells, intensity_img=intensity_img)
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
        elif feature in global_features:
            if feature == "total_area":
                data = self.calculate_total_area(valid_cells)
            elif feature == "density":
                data = self.calculate_density(frame, valid_cells, reference)
            elif feature == "type_fraction":
                data = self.calculate_type_fraction(frame, valid_cells, reference)
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


    def get_valid_non_edge_cells(self, frame, cells):
        labels = self.get_labels(frame)
        edge_cells_index = self.detect_edge_cells(labels)
        valid_cells = cells.query("valid == 1 and empty_cell == 0")
        return valid_cells[~valid_cells.index.isin(edge_cells_index)]

    def calculate_spatial_data(self, frame, window_radius, step_size, feature, cells_type='all'):
        labels = self.get_labels(frame)
        cells_info = self.get_cells_info(frame)
        res = np.zeros(labels.shape)
        valid_cells = self.get_valid_non_edge_cells(frame, cells_info)
        reference = None
        for y in range(step_size//2, res.shape[0], step_size):
            for x in range(step_size//2, res.shape[1], step_size):
                relevant_cells = self.get_cells_inside_a_circle(valid_cells, (y,x), window_radius)
                if feature == "density":
                    reference = np.sum(relevant_cells.area)
                elif feature == "type_fraction":
                    reference = relevant_cells.shape[0]
                if cells_type != 'all':
                    relevant_cells = relevant_cells.query("type == \"%s\"" % cells_type)
                # Calculate feature average
                if relevant_cells.shape[0] > 0:
                    data, err_msg = self.get_frame_data(frame, feature, relevant_cells,
                                                        special_features=self.SPECIAL_FEATURES + self.SPECIAL_X_ONLY_FEATURES,
                                                        global_features=self.GLOBAL_FEATURES,
                                                        for_histogram=True, reference=reference)
                else:
                    data, err_msg = 0, ""
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


    def plot_single_frame_data(self, frame, x_feature, y_feature, ax, cells_type='all', intensity_image=None):
        cell_info = self.get_cells_info(frame)
        y_data = None
        if cell_info is None:
            return None, "No frame data is available"
        if cells_type == "all":
            valid_cells = self.get_valid_non_edge_cells(frame, cell_info)
        else:
            cells_from_right_type = cell_info.query("type ==\"%s\"" % cells_type)
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

    def plot_spatial_map(self, frame, feature, window_radius, window_step, ax, cells_type='all', vmin=None, vmax=None):
        map, msg = self.calculate_spatial_data(frame, window_radius, window_step, feature, cells_type=cells_type)
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

    def plot_compare_frames_data(self, frames, feature, ax, cells_type='all'):
        data = []
        err = []
        n_results = []
        for frame in frames:
            cell_info = self.get_cells_info(frame)
            if cell_info is None:
                return None, "No frame data is available for frame %d" % frame
            if cells_type == "all":
                valid_cells = self.get_valid_non_edge_cells(frame, cell_info)
            else:
                cells_from_right_type = cell_info.query("type ==\"%s\"" % cells_type)
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
            ax.errorbar(frames, data, yerr = err, fmt="*")
        else:
            x_pos = np.arange(len(frames))
            x_labels = ["frame %d (N = %d)" % (f, n) for f,n in zip(frames, n_results)]
            ax.bar(x_pos, data, yerr=err, align='center', alpha=0.5, ecolor='black', capsize=10)
            ax.set_ylabel(feature)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)
            ax.yaxis.grid(True)
        title = "%s for different frames" % feature
        if cells_type != 'all':
            title += " for %s only" % cells_type
        ax.set_title(title)
        return pd.DataFrame({"Frame": frames, feature + " average": data, feature + " se": err, "N": n_results}), ""


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
            all_cells = cells_info.query("valid == 1")
            reference_cell_num = all_cells.shape[0]
        if reference_cell_num > 0:
            return relevant_cells.shape[0] / reference_cell_num
        else:
            return 0

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
                if np.isnan(self.drifts[frame - 1,0]):
                    self.drifts[frame - 1, :] = np.nan
                    update_next_drift = True
                continue
            if use_existing_drifts and not update_next_drift:
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
            cells_info.loc[indices_in_current_frame, "label"] = labels_previous_frame
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


    def calc_cell_types(self, hc_marker_image, frame_number, properties=None, hc_threshold=0.1,
                        percentage_above_threshold=90):
        cells_info = self.get_cells_info(frame_number)
        labels = self.get_labels(frame_number)
        self.get_cell_types(frame_number)
        max_brightness = np.percentile(hc_marker_image, 99)
        cell_types = np.zeros(labels.shape)
        if properties is None:
            for cell_index in range(len(cells_info)):
                if cells_info.valid[cell_index] == 1:
                    cell_pixels = hc_marker_image[labels == cell_index]
                    percentile_cell_brightness = np.percentile(cell_pixels, 100-percentage_above_threshold)
                    if percentile_cell_brightness > hc_threshold*max_brightness:
                        self.cells_info.at[cell_index, "type"] = "HC"
                        cell_types[labels == cell_index] = HC_TYPE
                    else:
                        self.cells_info.at[cell_index, "type"] = "SC"
                        cell_types[labels == cell_index] = SC_TYPE
        else:
            cell_indices = properties['label'] - 1
            threshold = hc_threshold * max_brightness
            atoh_intensities = properties["percentile_intensity"]
            HC_indices = cell_indices[atoh_intensities > threshold]
            SC_indices = cell_indices[atoh_intensities <= threshold]
            self.cells_info.loc[HC_indices, "type"] = "HC"
            self.cells_info.loc[SC_indices, "type"] = "SC"
            cell_types[np.isin(labels, HC_indices+1)] = HC_TYPE
            cell_types[np.isin(labels, SC_indices+1)] = SC_TYPE
        invalid_cells = np.argwhere(self.cells_info.valid.to_numpy() == 0).flatten()
        self.cells_info.loc[invalid_cells, "type"] = "invalid"
        cell_types[np.isin(labels, invalid_cells+1)] = INVALID_TYPE
        self.set_cell_types(frame_number, cell_types)

    def find_second_order_neighbors(self, frame, cells=None, cell_type='all'):
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
                    second_neighbors_info = cell_info.iloc[second_neighbors - 1]
                    valid_neighbors_info = second_neighbors_info.query("valid == 1 and type == \"%s\"" % cell_type)
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
        if cell is None or cell.empty_cell.values[0] == 1:
            return img
        else:
            rr, cc = disk((cell.cy.values[0], cell.cx.values[0]), radius, shape=img.shape)
            img[rr, cc] = 1
        return img[np.newaxis, :,:] * np.array(TRACK_COLOR).reshape((3,1,1))

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
                              hc_threshold=0.1, use_existing_types=False, percentile_above_threshold=90):
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
                                                           hc_threshold, use_existing_types, percentile_above_threshold)
            self._finished_last_line_addition = True
        return int(points_too_far)

    def remove_segmentation_line(self, frame, point1, hc_marker_image=None, hc_threshold=0.1, part_of_undo=False,
                                 use_existing_types=False, percentage_above_threshold=90):
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
                                                    hc_marker_image, hc_threshold, part_of_undo, use_existing_types,
                                                    percentage_above_threshold)
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
            try:
                current_type = cells_info.type[cell_idx]
                new_type = "SC" if (current_type == HC_TYPE) else "HC"
                self.cells_info.at[cell_idx, "type"] = new_type
                if current_type == "invalid":
                    self.cells_info.at[cell_idx, "valid"] = 1
            except IndexError:
                return 0
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
            try:
                self.cells_info.at[cell_idx, "type"] = "invalid"
                self.cells_info.at[cell_idx, "valid"] = 0
            except IndexError:
                return 0
        cell_types = self.get_cell_types(frame)
        if cell_types is not None:
            self.cell_types[labels == cell_idx + 1] = INVALID_TYPE
        return 0

    def update_after_segmentation_line_removal(self, cell1_label, cell2_label, frame, hc_marker_image=None,
                                               hc_threshold=0.1, part_of_undo=False, use_existing_types=False,
                                               percentage_above_threshold=90):
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
                        mean_area = np.mean(cell_info.area.to_numpy())
                        max_area = self.max_cell_area * mean_area
                        min_area = self.min_cell_area * mean_area
                        valid = min_area < area1 + area2 < max_area
                        cell_info.at[new_label - 1, "valid"] = valid
                        if use_existing_types and cell_types is not None:
                            hc_marker_image = (cell_types == HC_TYPE).astype(float)
                        if hc_marker_image is not None and not use_existing_types:
                            if valid:
                                percentage_intensity = np.percentile(hc_marker_image[labels == new_label],
                                                                     100-percentage_above_threshold)
                                if percentage_intensity > hc_threshold * np.percentile(hc_marker_image[hc_marker_image>0], 99):
                                    cell_info.at[new_label - 1, "type"] = "HC"
                                    if cell_types is not None:
                                        cell_types[labels == new_label] = HC_TYPE
                                else:
                                    cell_info.at[new_label - 1, "type"] = "SC"
                                    if cell_types is not None:
                                        cell_types[labels == new_label] = SC_TYPE
                            else:
                                cell_info.at[new_label - 1, "type"] = "invalid"
                                if cell_types is not None:
                                    cell_types[labels == new_label] = INVALID_TYPE
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

    def update_after_adding_segmentation_line(self, cell_label, frame, hc_marker_image=None, hc_threshold=0.1,
                                              use_existing_types=False, percentage_above_threshold=90):
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
                    if use_existing_types and cell_types is not None:
                        hc_marker_image = (cell_types == HC_TYPE).astype(float)
                    if hc_marker_image is None:
                        properties = regionprops(cell_region)
                    else:
                        hc_marker_region = hc_marker_image[region_first_row:region_last_row,
                                                           region_first_col:region_last_col]

                        def percentile_intensity(regionmask, intensity):
                            return np.percentile(intensity[regionmask], 100-percentage_above_threshold)
                        properties = regionprops_table(cell_region, intensity_image=hc_marker_region,
                                                       properties=("label", "area", "perimeter", "centroid", "bbox"),
                                                       extra_properties=(percentile_intensity,))
                        max_intensity = np.percentile(hc_marker_image[hc_marker_image > 0], 99)
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
                            if hc_marker_image is not None:
                                if cell_valid:
                                    percentage_intensity = properties["percentile_intensity"][region_index]
                                    if percentage_intensity > hc_threshold * max_intensity:
                                        cell_info.at[cell_label - 1, "type"] = "HC"
                                        if cell_types is not None:
                                            cell_types[labels == cell_label] = HC_TYPE
                                    else:
                                        cell_info.at[cell_label - 1, "type"] = "SC"
                                        if cell_types is not None:
                                            cell_types[labels == cell_label] = SC_TYPE
                                else:
                                    cell_info.at[cell_label - 1, "type"] = "invalid"
                                    if cell_types is not None:
                                        cell_types[labels == cell_label] = INVALID_TYPE
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
                                             "n_neighbors": 0}
                            if hc_marker_image is not None:
                                if new_cell_valid:
                                    percentage_intensity = properties["percentile_intensity"][region_index]
                                    if percentage_intensity > hc_threshold * max_intensity:
                                        new_cell_info["type"] = "HC"
                                        if cell_types is not None:
                                            cell_types[labels == new_label] = HC_TYPE
                                    else:
                                        new_cell_info["type"] = "SC"
                                        if cell_types is not None:
                                            cell_types[labels == new_label] = SC_TYPE
                                else:
                                    new_cell_info["type"] = "invalid"
                                    if cell_types is not None:
                                        cell_types[labels == new_label] = INVALID_TYPE
                            cell_info.loc[new_label - 1] = pd.Series(new_cell_info)
                    old_cell_neighbors = list(cell_info.neighbors[cell_label - 1].copy())
                    for neighbor_label in old_cell_neighbors:
                        cell_info.at[neighbor_label - 1, "neighbors"].remove(cell_label)
                    cell_info.at[cell_label - 1, "neighbors"] = set()
                    need_to_update_neighbors = list(cell_info.neighbors[cell_label]) + [cell_label, new_label]
                    self.find_neighbors(frame, labels_region=cell_region, only_for_labels=need_to_update_neighbors)
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

    def remove_labels(self):
        file_path = os.path.join(self.working_dir, "frame_%d_labels.npy" % self.labels_frame)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except OSError as e:
            print(str(e))
            sleep(1)
            self.remove_labels()

    def remove_cell_types(self):
        file_path = os.path.join(self.working_dir, "frame_%d_types.npy" % self.cell_types_frame)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except OSError as e:
            print(str(e))
            sleep(1)
            self.remove_cell_types()

    def remove_cells_info(self):
        file_path = os.path.join(self.working_dir, "frame_%d_data.pkl" % self.cells_info_frame)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except OSError as e:
            print(str(e))
            sleep(1)
            self.remove_cells_info()

    def save_labels(self):
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
        if self.drifts is not None:
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
        rmtree(old_working_dir)
        if self.labels_frame > 0:
            self.load_labels(self.labels_frame)
        if self.cell_types_frame > 0:
            self.load_cell_types(self.cell_types_frame)
        if self.cells_info_frame > 0:
            self.load_cells_info(self.cells_info_frame)
        self.load_events()
        self.load_drifts()
        self.load_valid_frames()
        self.load_shape_fitting()
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
        self.save_labels()
        self.save_cell_types()
        self.save_cells_info()


