import logging

import numpy as np
import pandas as pd
from PyQt5 import QtCore, uic, QtWidgets, QtGui #, QtWebEngineWidgets
import tissue_info
import matplotlib
import pickle
matplotlib.use('Qt5Agg')
import os.path, shutil, sys
import re
from basic_image_manipulations import *
from tissue_info import Tissue, INVALID_TYPE_INDEX
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import cv2
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import time
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager
from IPython.lib import guisupport

COLORTABLE=[]
WORKING_DIR = "D:\\Users\\TelAvivU-Analysis2\\Desktop\\Tomer\\Movies"
BASEDIR = os.path.dirname(__file__)
from tissue_info import INVALID_TYPE_INDEX

sys.path.insert(1, os.path.join(BASEDIR, '..\\Segmentation'))   # Tomer changed to this

from prediction_local import SegmentationPredictor
UNET_WEIGHTS_PATH = os.path.join('D:\\Users\\TelAvivU-Analysis2\\Desktop\\Tomer\\tissue_image_processing\\Segmentation',
                                   'model_image_segmentation_run_after_load_weights_large_dataset_3_ADAM.h5')
from numexpr import utils

utils.MAX_THREADS = 8

FIX_SEGMENTATION_OFF = 0
FIX_SEGMENTATION_ON = 1
FIX_SEGMENTATION_LINE = 2


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class CustomNavigationToolbar(NavigationToolbar):
    def __init__(self, canvas, parent, data, working_dir=None):
        NavigationToolbar.__init__(self, canvas, parent)
        self.data = data
        self.working_dir = working_dir

    def save_figure(self, *args):
        if self.working_dir is not None:
            matplotlib.rcParams['savefig.directory'] = self.working_dir
        super(CustomNavigationToolbar, self).save_figure(*args)
        if self.data is not None:
            file_path = QtWidgets.QFileDialog.getSaveFileName(self, 'Choose a file name to save data',
                                                    directory=matplotlib.rcParams['savefig.directory'])[0]
            if file_path:
                if file_path.endswith(".csv"):
                    if isinstance(self.data, pd.DataFrame):
                        self.data.to_csv(file_path)
                    elif isinstance(self.data, np.ndarray):
                        df = pd.DataFrame(self.data)
                        df.to_csv(file_path)
                    elif isinstance(self.data, dict):
                        with open(file_path, 'wb') as f:
                            import csv
                            with open(file_path, 'w') as output:
                                writer = csv.writer(output)
                                for key, value in self.data.items():
                                    writer.writerow([key, value])
                else:
                    if isinstance(self.data, pd.DataFrame):
                        self.data.to_pickle(file_path)
                    elif isinstance(self.data, np.ndarray):
                        np.save(file_path, self.data)
                    elif isinstance(self.data, dict):
                        with open(file_path, 'wb') as f:
                            pickle.dump(self.data, f)

class PlotDataWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None, working_dir=None):
        super(PlotDataWindow, self).__init__(parent)
        self.setWindowTitle("Data plot")
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.ax = self.canvas.axes
        self.working_dir = working_dir

    def create_toolbar(self, data):
        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = CustomNavigationToolbar(self.canvas, self, data, self.working_dir)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def get_ax(self):
        return self.ax

    def show(self, data=None):
        self.create_toolbar(data)
        super(PlotDataWindow, self).show()


class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        QtCore.QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[col]
        return None


class ConsoleWidget(RichJupyterWidget):
    def __init__(self, customBanner=None, *args, **kwargs):
        super(ConsoleWidget, self).__init__(*args, **kwargs)

        if customBanner is not None:
            self.banner = customBanner

        self.font_size = 10
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel(show_banner=False)
        self.kernel_manager.kernel.log.setLevel(logging.CRITICAL)
        self.kernel_manager.kernel.gui = 'qt'
        self.kernel_client = self._kernel_manager.client()
        self.kernel_client.start_channels()

        def stop():
            self.kernel_client.stop_channels()
            self.kernel_manager.shutdown_kernel()
            guisupport.get_app_qt4().exit()

        self.exit_requested.connect(stop)

    def push_vars(self, variableDict):
        """
        Given a dictionary containing name / value pairs, push those variables
        to the Jupyter console widget
        """
        self.kernel_manager.kernel.shell.push(variableDict)

    def clear(self):
        """
        Clears the terminal
        """
        self._control.clear()

    def print_text(self, text):
        """
        Prints some plain text to the console
        """
        self._append_plain_text(text)

    def execute_command(self, command):
        """
        Execute a command in the frame of the console widget
        """
        self._execute(command, False)

class AddTypeDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.inputs = []
        layout = QtWidgets.QFormLayout(self)
        self.inputs.append(QtWidgets.QLineEdit(self))
        layout.addRow("New type name:", self.inputs[0])
        self.inputs.append(QtWidgets.QLineEdit(self))
        layout.addRow("What channel do you want to display for the new type:", self.inputs[1])
        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        layout.addWidget(buttonBox)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def get_inputs(self):
        return (input.text() for input in self.inputs)

class ChannelsNameInputDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, input_num=1):
        super().__init__(parent)

        self.inputs = []
        layout = QtWidgets.QFormLayout(self)
        for i in range(input_num):
            self.inputs.append(QtWidgets.QLineEdit(self))
            layout.addRow("Channel %d name:" % (i + 1), self.inputs[i])
        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        layout.addWidget(buttonBox)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def get_inputs(self):
        return [input.text() for input in self.inputs]


class FormImageProcessing(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(FormImageProcessing, self).__init__(parent)
        uic.loadUi(os.path.join(BASEDIR, "movie_display.ui"), self)
        self.setWindowTitle("Movie Segmentation")
        self.setState()
        self.saveFile = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        self.undo = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self)
        self.next = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self)
        self.previous = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self)
        self.toggle_valid = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_V), self)
        self.connect_methods()
        self.hide_progress_bars()
        self.img = None
        self.current_frame_number = 0
        self.img_in_memory = False
        self.img_dimensions = None
        self.img_metadata = None
        self.current_frame = None
        self.current_segmentation = None
        self.current_analysis_image = None
        self.current_labels = None
        self.zo_changed = False
        self.atoh_changed = False
        self.segmentation_changed = False
        self.analysis_changed = False
        self.tissue_info = None
        self.segmentation_saved = True
        self.number_of_frames = 0
        self.number_of_channels = 0
        self.channel_names = []
        self.fake_channels = []
        self.analysis_saved = True
        self.waiting_for_data_save = []
        self.fixing_segmentation_mode = FIX_SEGMENTATION_OFF
        self.fix_segmentation_last_position = None
        self.fix_cell_types_on = False
        self.fix_tracking_on = False
        self.mark_event_stage = 0
        self.event_start_frame = None
        self.event_end_frame = None
        self.event_start_position = None
        self.event_end_position = None
        self.working_directory = WORKING_DIR
        self.epyseg_dir = None
        self.epyseg = None
        self.fitting_stage = 0

    def closeEvent(self, event):
        if self.tissue_info is None:
            event.accept()
        elif self.data_lost_warning(self.close):
            self.tissue_info.clean_up()
            event.accept()  # let the window close
        else:
            event.ignore()

    def close(self):
        super(FormImageProcessing, self).close()

    def open_console(self):
        window = QtWidgets.QMainWindow(parent=self)
        console = ConsoleWidget(parent=window)
        vars = {"img": self.img, "img_dimensions": self.img_dimensions, "current_frame": self.current_frame,
                "tissue_info": self.tissue_info, "number_of_frames": self.number_of_frames,
                "image_in_memory":self.img_in_memory, "working_directory":self.working_directory,
                "img_metadata":self.img_metadata}
        console.push_vars(vars)
        window.resize(1500, 600)
        console.resize(1500, 600)
        window.show()


    def connect_methods(self):
        self.open_console_push_button.clicked.connect(self.open_console)
        self.zo_check_box.stateChanged.connect(self.zo_related_widget_changed)
        self.atoh_check_box.stateChanged.connect(self.atoh_related_widget_changed)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        self.frame_line_edit.editingFinished.connect(self.frame_line_edit_changed)
        self.zo_spin_box.valueChanged.connect(self.zo_related_widget_changed)
        self.atoh_spin_box.valueChanged.connect(self.atoh_related_widget_changed)
        self.zo_level_scroll_bar.valueChanged.connect(self.zo_related_widget_changed)
        self.atoh_level_scroll_bar.valueChanged.connect(self.atoh_related_widget_changed)
        self.segment_frame_button.clicked.connect(self.segment_frame)
        self.analyze_frame_button.clicked.connect(self.find_cell_types_in_frame)
        self.open_file_pb.clicked.connect(self.open_file)
        self.segment_all_frames_button.clicked.connect(self.segment_all_frames)
        self.show_segmentation_check_box.stateChanged.connect(self.segmentation_related_widget_changed)
        self.show_cell_types_check_box.stateChanged.connect(self.analysis_related_widget_changed)
        self.show_cell_tracking_check_box.stateChanged.connect(self.cell_tracking_check_box_changed)
        self.cell_tracking_spin_box.valueChanged.connect(self.analysis_related_widget_changed)
        self.show_neighbors_check_box.stateChanged.connect(self.analysis_related_widget_changed)
        self.show_events_check_box.stateChanged.connect(self.analysis_related_widget_changed)
        self.show_events_button.clicked.connect(self.display_events)
        self.saveFile.activated.connect(self.save_data)
        self.undo.activated.connect(self.undo_last_action)
        self.next.activated.connect(self.next_frame)
        self.previous.activated.connect(self.previous_frame)
        self.toggle_valid.activated.connect(self.toggle_valid_frame)
        self.save_segmentation_button.clicked.connect(self.save_data)
        self.export_to_tiff_button.clicked.connect(self.export_segmentation_to_tiff)
        self.load_segmentation_button.clicked.connect(self.load_data)
        self.analyze_segmentation_button.clicked.connect(self.analyze_segmentation)
        self.track_cells_button.clicked.connect(self.track_cells)
        self.find_events_button.clicked.connect(self.find_events)
        self.cancel_segmentation_button.clicked.connect(self.cancel_segmentation)
        self.cancel_analysis_button.clicked.connect(self.cancel_analysis)
        self.cancel_tracking_button.clicked.connect(self.cancel_tracking)
        self.cancel_event_finding_button.clicked.connect(self.cancel_event_finding)
        self.plot_single_cell_data_button.clicked.connect(self.plot_single_cell_data)
        self.plot_event_related_data_push_button.clicked.connect(self.plot_event_related_data)
        self.plot_single_frame_data_button.clicked.connect(self.plot_single_frame_data)
        self.plot_spatial_map_button.clicked.connect(self.plot_spatial_map)
        self.plot_compare_frames_button.clicked.connect(self.plot_compare_frames_data)
        self.image_display.photoClicked.connect(self.image_clicked)
        self.fix_segmentation_button.clicked.connect(self.fix_segmentation)
        self.finish_fixing_segmentation_button.clicked.connect(self.finish_fixing_segmentation)
        self.fix_cell_types_button.clicked.connect(self.fix_cell_types)
        self.finish_fixing_cell_types_button.clicked.connect(self.finish_fixing_cell_types)
        self.remove_non_sensory_button.clicked.connect(self.remove_nonsensory_cells)
        self.fix_tracking_button.clicked.connect(self.correct_tracking)
        self.mark_event_button.clicked.connect(self.mark_event)
        self.delete_events_button.clicked.connect(self.delete_events)
        self.mark_event_combo_box.clear()
        self.mark_event_combo_box.addItems(tissue_info.Tissue.EVENT_TYPES)
        self.mark_event_combo_box.addItems(tissue_info.Tissue.ADDITIONAL_EVENT_MARKING_OPTION)
        self.abort_event_marking_button.clicked.connect(self.abort_event_marking)
        self.valid_frame_check_box.stateChanged.connect(self.change_frame_validity)
        self.cell_size_spin_box_min.valueChanged.connect(self.update_min_max_cell_area)
        self.cell_size_spin_box_max.valueChanged.connect(self.update_min_max_cell_area)
        self.fit_a_shape_button.clicked.connect(self.fit_a_shape)
        self.finish_fitting_a_shape_button.clicked.connect(self.finish_fitting_a_shape)
        self.event_type_combo_box.clear()
        self.event_type_combo_box.addItems(tissue_info.Tissue.EVENT_TYPES)
        self.event_type_combo_box.addItems(tissue_info.Tissue.ADDITIONAL_EVENT_STATISTICS_OPTIONS)
        self.plot_event_statistics_botton.clicked.connect(self.plot_event_statistics)
        self.add_type_button.clicked.connect(self.add_type)

    def open_file(self):
        global img
        if self.tissue_info is not None:
            if not self.data_lost_warning(self.open_file):
                return 0
            else:
                self.tissue_info.clean_up()
                self.tissue_info = None
        fname = QtWidgets.QFileDialog.getOpenFileName(caption='Open File',
                                            directory=self.working_directory, filter="images (*.czi, *.tif)")[0]
        if not fname or os.path.isdir(fname):
            return 0
        try:
            if self.load_to_memory_check_box.isChecked():
                self.img, self.img_dimensions, self.img_metadata = read_whole_image(fname)
                self.img_in_memory = True
            else:
                self.img, self.img_dimensions, self.img_metadata = read_virtual_image(fname)
                self.img_in_memory = False
        except PermissionError or ValueError:
            message_box = QtWidgets.QMessageBox
            message_box.about(self,'', 'An error has occurd while oppening file %s' % fname)
            return 0
        self.working_directory = os.path.dirname(fname)
        self.number_of_frames = self.img_dimensions.T
        self.number_of_channels = self.img_dimensions.C
        dialog = ChannelsNameInputDialog(self, self.number_of_channels)
        if dialog.exec():
            self.channel_names = dialog.get_inputs()
        else:
            self.channel_names = [str(i + 1) for i in range(self.number_of_channels)]
        zo_channel_name = self.channel_names[self.zo_spin_box.value()]
        self.zo_check_box.setText("Seg channel: %s" % zo_channel_name)
        self.zo_level_label.setText("%s level" % zo_channel_name)
        self.zo_changed = True
        atoh_channel_name = self.channel_names[self.atoh_spin_box.value()]
        self.atoh_check_box.setText("Type channel: %s" % atoh_channel_name)
        self.atoh_level_label.setText("%s level" % atoh_channel_name)
        self.atoh_changed = True
        self.atoh_spin_box.setMaximum(self.img_dimensions.C-1)
        self.zo_spin_box.setMaximum(self.img_dimensions.C-1)
        self.frame_slider.setMaximum(self.number_of_frames)
        self.choose_reference_frame_spin_box.setMaximum(self.number_of_frames)
        self.current_frame = np.zeros((3, self.img_dimensions.X, self.img_dimensions.Y), dtype="uint8")
        max_cell_area = self.cell_size_spin_box_max.value() / 100
        min_cell_area = self.cell_size_spin_box_min.value() / 100
        if self.tissue_info is not None:
            self.tissue_info.clean_up()
        self.tissue_info = Tissue(self.number_of_frames, fname, self.channel_names, max_cell_area=max_cell_area,
                                  min_cell_area=min_cell_area, load_to_memory=self.img_in_memory)
        self.frame_line_edit.setText("%d/%d" % (self.frame_slider.value(), self.number_of_frames))
        self.setWindowTitle(fname)
        self.current_frame_number = 1
        self.display_frame()
        min_planar_dimension = min(self.img_dimensions.X, self.img_dimensions.Y)
        self.window_radius_spin_box.setMaximum(min_planar_dimension)
        self.spatial_resolution_spin_box.setMaximum(min_planar_dimension)
        self.event_statistics_window_radius_x_data_label_spin_box.setMaximum(min_planar_dimension)
        self.event_statistics_window_radius_y_data_label_spin_box.setMaximum(min_planar_dimension)
        self.setState(image=True)

    def display_frame(self):
        former_frame = self.current_frame_number
        frame_number = self.frame_slider.value()
        self.current_frame_number = frame_number
        if self.zo_changed:
            if self.zo_check_box.isChecked():
                zo_channel = self.zo_spin_box.value()
                if zo_channel >= self.number_of_channels:
                    zo_channel = self.fake_channels[zo_channel - self.number_of_channels]
                if self.img_in_memory:
                    disp_img = self.img[frame_number - 1, zo_channel, 0, :, :].T
                else:
                    disp_img = self.img[frame_number - 1, zo_channel, 0, :, :].compute().T
                disp_img = disp_img * self.zo_level_scroll_bar.value() * (10 / max(np.average(disp_img),1))
                np.putmask(disp_img, disp_img > 255, 255)
                self.current_frame[1, :, :] = disp_img
            else:
                self.current_frame[1, :, :] = 0
            self.zo_changed = False
        if self.atoh_changed:
            if self.atoh_check_box.isChecked():
                atoh_channel = self.atoh_spin_box.value()
                if atoh_channel >= self.number_of_channels:
                    atoh_channel = self.fake_channels[atoh_channel - self.number_of_channels]
                if self.img_in_memory:
                   disp_img = self.img[frame_number - 1, atoh_channel, 0, :, :].T
                else:
                   disp_img = self.img[frame_number - 1, atoh_channel, 0, :, :].compute().T
                disp_img = self.atoh_level_scroll_bar.value() * disp_img * (10 / max(np.average(disp_img),1))
                np.putmask(disp_img, disp_img > 255, 255)
                self.current_frame[2, :, :] = disp_img
            else:
                self.current_frame[2, :, :] = 0
            self.atoh_changed = False
        if self.segmentation_changed:
            if self.show_segmentation_check_box.isChecked() and self.current_segmentation is not None:
                self.current_frame[0, :, :] = 255*self.current_segmentation
            else:
                self.current_frame[0, :, :] = 0
            self.segmentation_changed = False
        add_analysis = False
        analysis_img = None
        if self.analysis_changed:
            analysis_img = self.get_analysis_img(self.show_cell_types_check_box.isChecked(),
                                                 self.show_neighbors_check_box.isChecked(),
                                                 self.show_cell_tracking_check_box.isChecked(),
                                                 self.cell_tracking_spin_box.value(),
                                                 self.show_events_check_box.isChecked(),
                                                 self.fitting_stage > 0)
            if analysis_img is not None:
                add_analysis = True
        current_frame_display = self.current_frame.copy()
        current_frame_display[1][self.current_frame[0] > 0] = 0
        current_frame_display[2][self.current_frame[0] > 0] = 0
        if add_analysis:
            disp_image = np.transpose(np.where(analysis_img == 0, current_frame_display,
                                              np.round(analysis_img*255).astype("uint8")), (1, 2, 0))
        else:
            disp_image = np.transpose(current_frame_display, (1, 2, 0))
        disp_image = cv2.cvtColor(disp_image, cv2.COLOR_BGR2RGB)
        QI = QtGui.QImage(bytes(disp_image), self.img_dimensions.Y, self.img_dimensions.X, 3*self.img_dimensions.Y, QtGui.QImage.Format_RGB888)
        QI.setColorTable(COLORTABLE)
        # if former_frame < frame_number:
        #     translation = self.tissue_info.drifts[frame_number-1, :]
        #     if not np.isnan(translation).any():
        #         self.image_display.translate(translation[1], translation[0])
        # elif former_frame > frame_number:
        #     translation = -self.tissue_info.drifts[former_frame-1, :]
        #     if not np.isnan(translation).any():
        #         self.image_display.translate(translation[1], translation[0])
        self.image_display.setPhoto(QtGui.QPixmap.fromImage(QI))
        self.display_histogram()
        valid_frame = self.tissue_info.is_valid_frame(frame_number)
        self.valid_frame_check_box.setChecked(valid_frame)

    def next_frame(self):
        next_valid_frame = self.frame_slider.value() + 1
        if next_valid_frame > self.number_of_frames:
            return 0
        while not self.tissue_info.is_valid_frame(next_valid_frame):
            next_valid_frame += 1
            if next_valid_frame > self.number_of_frames:
                return 0
        self.frame_slider.setValue(next_valid_frame)

    def previous_frame(self):
        previous_valid_frame = self.frame_slider.value() - 1
        if previous_valid_frame < 1:
            return 0
        while not self.tissue_info.is_valid_frame(previous_valid_frame):
            previous_valid_frame -= 1
            if previous_valid_frame < 1:
                return 0
        self.frame_slider.setValue(previous_valid_frame)

    def toggle_valid_frame(self):
        self.valid_frame_check_box.setChecked(not self.valid_frame_check_box.isChecked())


    def display_histogram(self):
        channel = self.img_dimensions.C
        if channel == 1:  # grayscale image
            mask = np.logical_and(self.current_frame[0] > 0, self.current_frame[0] < 255).astype("uint8")
            histr = cv2.calcHist(self.current_frame, [0], mask, [256], [0, 256]).flatten()
            self.histogram_widget.plot(histr,pen='y', linewidth=3.0)

        else:  # color image
            color = ('b', 'g', 'r')
            for i in range(3):
                col = color[i]
                mask = np.logical_and(self.current_frame[i] > 0, self.current_frame[i] < 255).astype("uint8")
                histr = cv2.calcHist(self.current_frame, [i], mask, [256], [0, 256]).flatten()
                self.histogram_widget.plot(histr,pen=col, linewidth=3.0)
        self.histogram_widget.setLabel('left','Frequency')
        self.histogram_widget.setLabel('bottom','Intensity')
        self.histogram_widget.update()


    def frame_changed(self):
        self.zo_changed = True
        self.atoh_changed = True
        self.segmentation_changed = True
        self.current_segmentation = self.tissue_info.get_segmentation(self.frame_slider.value())
        self.pixel_info.setText('Press the image to get pixel info')
        self.display_frame()
        self.cells_number_changed()
        self.update_shape_fitting()
        if self.current_segmentation is None:
            self.show_segmentation_check_box.setEnabled(False)

    def change_frame_validity(self):
        frame_number = self.frame_slider.value()
        valid = self.valid_frame_check_box.isChecked()
        self.tissue_info.set_validity_of_frame(frame_number, valid)

    def cells_number_changed(self):
        cells_num = self.tissue_info.get_cells_number()
        self.cell_tracking_spin_box.setMaximum(int(cells_num))
        self.plot_single_cell_data_spin_box.setMaximum(int(cells_num))

    def update_min_max_cell_area(self):
        if self.tissue_info is not None:
            max_cell_area = self.cell_size_spin_box_max.value() / 100
            min_cell_area = self.cell_size_spin_box_min.value() / 100
            self.tissue_info.set_valid_cell_area(min_cell_area, max_cell_area)

    def update_single_cell_features(self):
        features = self.tissue_info.get_cells_features(self.frame_slider.value())
        self.plot_single_cell_data_combo_box.clear()
        self.plot_single_cell_data_combo_box.addItems(features)
        self.plot_single_cell_data_combo_box.addItems(self.tissue_info.SPATIAL_FEATURES)

    def update_single_frame_features(self):
        features = self.tissue_info.get_cells_features(self.frame_slider.value())
        self.plot_single_frame_data_x_combo_box.clear()
        self.plot_single_frame_data_x_combo_box.addItems(features)
        self.plot_single_frame_data_x_combo_box.addItems(self.tissue_info.SPECIAL_FEATURES)
        self.plot_single_frame_data_x_combo_box.addItems(self.tissue_info.SPECIAL_X_ONLY_FEATURES)
        self.compare_frames_combo_box.clear()
        self.compare_frames_combo_box.addItems(features)
        self.compare_frames_combo_box.addItems(self.tissue_info.SPECIAL_FEATURES)
        self.compare_frames_combo_box.addItems(self.tissue_info.SPECIAL_X_ONLY_FEATURES)
        self.compare_frames_combo_box.addItems(self.tissue_info.SPECIAL_Y_ONLY_FEATURES)
        self.compare_frames_combo_box.addItems(self.tissue_info.GLOBAL_FEATURES)
        self.plot_single_frame_data_y_combo_box.clear()
        self.plot_single_frame_data_y_combo_box.addItems(features)
        self.plot_single_frame_data_y_combo_box.addItems(self.tissue_info.SPECIAL_FEATURES)
        self.plot_single_frame_data_y_combo_box.addItems(self.tissue_info.SPECIAL_Y_ONLY_FEATURES)
        self.plot_single_frame_data_cell_type_combo_box.clear()
        self.plot_single_frame_data_cell_type_combo_box.addItems(self.tissue_info.get_cell_type_names())
        self.plot_compare_frame_data_cell_type_combo_box.clear()
        self.plot_compare_frame_data_cell_type_combo_box.addItems(self.tissue_info.get_cell_type_names())
        self.event_statistics_x_data_combo_box.clear()
        self.event_statistics_x_data_combo_box.addItems(features)
        self.event_statistics_x_data_combo_box.addItems(self.tissue_info.SPECIAL_FEATURES)
        self.event_statistics_x_data_combo_box.addItems(self.tissue_info.GLOBAL_FEATURES)
        self.event_statistics_x_data_combo_box.addItems(self.tissue_info.SPATIAL_FEATURES)
        self.event_statistics_x_data_combo_box.addItems(self.tissue_info.SPECIAL_EVENT_STATISTICS_FEATURES)
        self.event_statistics_y_data_combo_box.clear()
        self.event_statistics_y_data_combo_box.addItems(features)
        self.event_statistics_y_data_combo_box.addItems(self.tissue_info.SPECIAL_FEATURES)
        self.event_statistics_y_data_combo_box.addItems(self.tissue_info.GLOBAL_FEATURES)
        self.event_statistics_y_data_combo_box.addItems(self.tissue_info.SPATIAL_FEATURES)
        self.event_statistics_y_data_combo_box.addItems(["None"])


    def update_shape_fitting(self):
        shape_fitting_results = self.tissue_info.get_shape_fitting_results(self.frame_slider.value())
        new_features = ["%s:%s" % (shape_name, key) for shape_name in shape_fitting_results.keys()
                        for key in shape_fitting_results[shape_name].keys()]
        for feature in new_features:
            if self.compare_frames_combo_box.findText(feature) == -1:
                self.compare_frames_combo_box.addItem(feature)


    def mark_event(self, pos=None):
        self.mark_event_button.setEnabled(False)
        self.mark_event_button.hide()
        self.delete_events_button.setEnabled(False)
        self.abort_event_marking_button.setEnabled(True)
        self.abort_event_marking_button.show()
        self.mark_event_combo_box.setEnabled(False)
        if self.mark_event_combo_box.currentIndex() < 0:
            return 0
        if self.mark_event_stage == 0:
            message_box = QtWidgets.QMessageBox
            message_box.about(self, '', 'Go to the first frame of the event\nand click on the relevant cell')
            self.mark_event_stage = 1
        elif self.mark_event_stage == 1:
            self.event_start_position = pos
            self.event_start_frame = self.frame_slider.value()
            if self.mark_event_combo_box.currentText() == "delete event":
                self.tissue_info.add_event(self.mark_event_combo_box.currentText(), self.event_start_frame,
                                           0, self.event_start_position, (0,0), source='manual')
                self.mark_event_stage = 0
            else:
                message_box = QtWidgets.QMessageBox
                message_box.about(self, '',
                                  'Go to the last frame of the event\nand click on the relevant cell\n(or cells, in case of division)')
                cell_id = self.tissue_info.get_cell_id_by_position(self.event_start_frame, self.event_start_position)
                if cell_id > 0:
                    self.show_cell_tracking_check_box.setChecked(True)
                    self.cell_tracking_spin_box.setValue(cell_id)
                self.mark_event_stage = 2

        elif self.mark_event_stage == 2:
            self.event_end_position = pos
            self.event_end_frame = self.frame_slider.value()
            if self.mark_event_combo_box.currentText() == "division":
                self.mark_event_stage = 3
            else:
                self.tissue_info.add_event(self.mark_event_combo_box.currentText(), self.event_start_frame,
                                           self.event_end_frame, self.event_start_position, self.event_end_position,
                                           source='manual')
                self.mark_event_stage = 0
        elif self.mark_event_stage == 3:
            self.event_end_frame = self.frame_slider.value()
            self.tissue_info.add_event(self.mark_event_combo_box.currentText(), self.event_start_frame,
                                       self.event_end_frame, self.event_start_position, self.event_end_position,
                                       pos, source='manual')
            self.mark_event_stage = 0
        if self.mark_event_stage == 0:
            self.abort_event_marking_button.setEnabled(False)
            self.abort_event_marking_button.hide()
            self.mark_event_button.setEnabled(True)
            self.mark_event_button.show()
            self.delete_events_button.setEnabled(True)
            self.mark_event_combo_box.setEnabled(True)
            self.analysis_changed = True
            self.display_frame()
        return 0

    def add_type(self):
        dialog = AddTypeDialog(self)
        if dialog.exec():
            type_name, type_channel = dialog.get_inputs()
            self.fake_channels.append(int(type_channel) - 1)
            self.channel_names.append(type_name)
            self.tissue_info.add_fake_type(type_name, int(type_channel) - 1)
            self.atoh_spin_box.setMaximum(len(self.channel_names))
            self.zo_spin_box.setMaximum(len(self.channel_names))

    def abort_event_marking(self):
        self.mark_event_stage = 0
        self.abort_event_marking_button.setEnabled(False)
        self.abort_event_marking_button.hide()
        self.mark_event_button.setEnabled(True)
        self.mark_event_button.show()
        self.delete_events_button.setEnabled(True)
        self.mark_event_combo_box.setEnabled(True)
        return 0

    def delete_events(self):
        if not self.data_lost_warning(self.delete_events):
            return 0
        source = 'automatic' if self.delete_events_aout_detected_check_box.isChecked() else 'all'
        if self.delete_events_frame_check_box.isChecked():
            frame = self.frame_slider.value()
            self.tissue_info.delete_all_events_in_frame(frame, source=source)
        else:
            self.tissue_info.delete_all_events(source=source)
        return 0

    def slider_changed(self):
        text = "%d/%d" % (self.frame_slider.value(), self.number_of_frames)
        if self.frame_line_edit.text() != text:
            self.frame_line_edit.setText(text)
            self.frame_changed()



    def image_clicked(self, click_info):
        pos = click_info.point
        if pos.y() < 0 or pos.y() >= self.img_dimensions.X or pos.x() < 0 or pos.x() >= self.img_dimensions.Y:
            return 0
        button = click_info.button
        double_click = click_info.doubleClick
        if self.image_display.dragMode() == QtWidgets.QGraphicsView.NoDrag:
            if self.fixing_segmentation_mode == FIX_SEGMENTATION_ON:
                frame = self.frame_slider.value()
                if button == QtCore.Qt.LeftButton:
                    if double_click:
                        if self.fix_segmentation_last_position is not None:
                            self.fix_segmentation_last_position = None
                            if self.img_in_memory:
                                hc_marker_img = self.img[frame - 1, self.atoh_spin_box.value(), 0, :, :].T
                            else:
                                hc_marker_img = self.img[frame - 1, self.atoh_spin_box.value(), 0, :, :].compute().T
                            self.tissue_info.add_segmentation_line(frame, (pos.x(), pos.y()), final=True)
                    else:
                        points_too_far = self.tissue_info.add_segmentation_line(frame, (pos.x(), pos.y()),
                                                               self.fix_segmentation_last_position,
                                                               initial=(self.fix_segmentation_last_position is None))
                        if points_too_far:
                            self.fix_segmentation_last_position = None
                        else:
                            self.fix_segmentation_last_position = (pos.x(), pos.y())
                elif button == QtCore.Qt.MiddleButton:
                    if self.fix_segmentation_last_position is not None:
                        self.tissue_info.add_segmentation_line(frame, self.fix_segmentation_last_position, final=True)
                        self.fix_segmentation_last_position = None
                    self.tissue_info.remove_segmentation_line(frame, (pos.x(), pos.y()))
                self.segmentation_changed = True
                self.current_segmentation = self.tissue_info.get_segmentation(frame)
                self.display_frame()
            elif self.fix_cell_types_on:
                frame = self.frame_slider.value()
                if button == QtCore.Qt.LeftButton:
                    self.tissue_info.change_cell_type(frame, (pos.x(), pos.y()), type_name=self.channel_names[self.atoh_spin_box.value()])
                elif button == QtCore.Qt.MiddleButton:
                    self.tissue_info.make_invalid_cell(frame, (pos.x(), pos.y()))
                self.analysis_changed = True
                self.display_frame()
            elif self.fix_tracking_on:
                frame = self.frame_slider.value()
                new_label = self.cell_tracking_spin_box.value()
                self.tissue_info.fix_cell_label(frame, (pos.x(), pos.y()), new_label)
                self.tissue_info.fix_cell_id_in_events()
                self.analysis_changed = True
                self.correct_tracking(off=True)
                self.display_frame()
            elif self.mark_event_stage > 0:
                self.mark_event((pos.x(), pos.y()))
            elif self.fitting_stage > 0:
                self.fit_a_shape((pos.x(), pos.y()))
            if self.pixel_info.isEnabled():
                text = 'pixel info: x = %d, y = %d' % (pos.x(), pos.y())
                cell = self.tissue_info.get_cell_by_pixel(pos.x(), pos.y(), self.frame_slider.value())
                if cell is not None:
                    if cell.empty:
                        cell_id = 0
                    else:
                        cell_id = cell.label
                    text += '\ncell id = %d' % cell_id
                    if double_click:
                        self.cell_tracking_spin_box.setValue(cell_id)
                self.pixel_info.setText(text)

    def undo_last_action(self):
        if self.fixing_segmentation_mode == FIX_SEGMENTATION_ON:
            if self.tissue_info.undo_last_action(self.frame_slider.value()) > 0:
                self.segmentation_changed = True
                self.current_segmentation = self.tissue_info.get_segmentation(self.frame_slider.value())
                self.display_frame()

    def frame_line_edit_changed(self):
        try:
            frame_num = int(self.frame_line_edit.text().split("/")[0])
            text = "%d/%d" % (frame_num, self.number_of_frames)
            self.frame_line_edit.setText(text)
            if self.frame_slider.value() != frame_num:
                self.frame_slider.setValue(frame_num)
                self.frame_changed()
        except ValueError:
            self.frame_line_edit.setText("%d/%d" % (self.frame_slider.value(), self.number_of_frames))

    def zo_related_widget_changed(self):
        self.zo_changed = True
        channel_name = self.channel_names[self.zo_spin_box.value()]
        self.zo_check_box.setText("Seg channel: %s" % channel_name)
        self.zo_level_label.setText("%s level" % channel_name)
        self.display_frame()

    def atoh_related_widget_changed(self):
        self.atoh_changed = True
        channel_name = self.channel_names[self.atoh_spin_box.value()]
        self.atoh_check_box.setText("Type channel: %s" % channel_name)
        self.atoh_level_label.setText("%s level" % channel_name)
        self.display_frame()

    def segmentation_related_widget_changed(self):
        self.segmentation_changed = True
        self.display_frame()

    def cell_tracking_check_box_changed(self):
        if self.show_cell_tracking_check_box.isChecked():
            self.fix_tracking_button.setEnabled(True)
        else:
            self.fix_tracking_button.setEnabled(False)
        self.analysis_related_widget_changed()

    def analysis_related_widget_changed(self):
        self.analysis_changed = True
        self.display_frame()

    def hide_progress_bars(self):
        self.segment_all_frames_progress_bar.hide()
        self.analyze_segmentation_progress_bar.hide()
        self.save_data_progress_bar.hide()
        self.load_data_progress_bar.hide()
        self.track_cells_progress_bar.hide()
        self.find_events_progress_bar.hide()
        self.cancel_segmentation_button.hide()
        self.cancel_analysis_button.hide()
        self.cancel_tracking_button.hide()
        self.cancel_event_finding_button.hide()
        self.finish_fixing_segmentation_button.hide()
        self.fix_segmentation_label.hide()
        self.finish_fixing_cell_types_button.hide()
        self.remove_non_sensory_button.hide()
        self.fix_cell_types_label.hide()
        self.fix_tracking_label.hide()

    def setState(self, image=False, segmentation=False, analysis=False):
        if image:
            self.frame_slider.setEnabled(True)
            self.frame_line_edit.setEnabled(True)
            self.zo_check_box.setEnabled(True)
            self.atoh_check_box.setEnabled(True)
            self.zo_spin_box.setEnabled(True)
            self.atoh_spin_box.setEnabled(True)
            self.atoh_level_scroll_bar.setEnabled(True)
            self.zo_level_scroll_bar.setEnabled(True)
            self.segment_frame_button.setEnabled(True)
            self.segmentation_threshold_spin_box.setEnabled(True)
            self.segmentation_kernel_std_spin_box.setEnabled(True)
            self.segmentation_block_size_spin_box.setEnabled(True)
            self.segmentation_block_size_label.setEnabled(True)
            self.segmentation_threshold_label.setEnabled(True)
            self.segmentation_kernel_std_label.setEnabled(True)
            self.load_segmentation_button.setEnabled(True)
            self.pixel_info.setEnabled(True)
            self.valid_frame_check_box.setEnabled(True)
        else:
            segmentation = False
            analysis = False
            self.frame_slider.setEnabled(False)
            self.frame_line_edit.setEnabled(False)
            self.zo_check_box.setEnabled(False)
            self.atoh_check_box.setEnabled(False)
            self.zo_spin_box.setEnabled(False)
            self.atoh_spin_box.setEnabled(False)
            self.atoh_level_scroll_bar.setEnabled(False)
            self.zo_level_scroll_bar.setEnabled(False)
            self.segment_frame_button.setEnabled(False)
            self.analyze_frame_button.setEnabled(False)
            self.segmentation_threshold_spin_box.setEnabled(False)
            self.segmentation_kernel_std_spin_box.setEnabled(False)
            self.segmentation_block_size_spin_box.setEnabled(False)
            self.segmentation_block_size_label.setEnabled(False)
            self.segmentation_threshold_label.setEnabled(False)
            self.segmentation_kernel_std_label.setEnabled(False)
            self.load_segmentation_button.setEnabled(False)
            self.track_cells_button.setEnabled(False)
            self.find_events_button.setEnabled(False)
            self.plot_single_cell_data_spin_box.setEnabled(False)
            self.plot_single_cell_data_combo_box.setEnabled(False)
            self.plot_single_cell_data_button.setEnabled(False)
            self.plot_event_related_data_push_button.setEnabled(False)
            self.plot_single_frame_data_button.setEnabled(False)
            self.plot_spatial_map_button.setEnabled(False)
            self.spatial_resolution_spin_box.setEnabled(False)
            self.spatial_resolution_label.setEnabled(False)
            self.window_radius_spin_box.setEnabled(False)
            self.window_radius_label.setEnabled(False)
            self.plot_single_frame_data_x_combo_box.setEnabled(False)
            self.plot_single_frame_data_y_combo_box.setEnabled(False)
            self.plot_single_frame_data_cell_type_combo_box.setEnabled(False)
            self.plot_compare_frames_button.setEnabled(False)
            self.plot_compare_frame_data_cell_type_combo_box.setEnabled(False)
            self.compare_frames_line_edit.setEnabled(False)
            self.compare_frames_combo_box.setEnabled(False)
            self.pixel_info.setEnabled(False)
            self.finish_fixing_segmentation_button.setEnabled(False)
            self.fix_segmentation_label.setEnabled(False)
            self.fix_segmentation_button.setEnabled(False)
            self.finish_fixing_cell_types_button.setEnabled(False)
            self.remove_non_sensory_button.setEnabled(False)
            self.fix_cell_types_label.setEnabled(False)
            self.fix_cell_types_button.setEnabled(False)
            self.mark_event_button.setEnabled(False)
            self.delete_events_button.setEnabled(False)
            self.delete_events_frame_check_box.setEnabled(False)
            self.delete_events_aout_detected_check_box.setEnabled(False)
            self.mark_event_combo_box.setEnabled(False)
            self.show_events_check_box.setEnabled(False)
            self.show_events_button.setEnabled(False)
            self.valid_frame_check_box.setEnabled(False)
            self.fit_a_shape_button.setEnabled(False)
            self.choose_marking_target_combo_box.setEnabled(False)
            self.choose_marking_target_combo_box.clear()
            self.choose_marking_target_combo_box.setCurrentText("Choose marking target")
            self.choose_marking_target_combo_box.setCurrentIndex(-1)
            self.choose_fitting_shape_combo_box.clear()
            self.choose_fitting_shape_combo_box.setCurrentText("Choose shape")
            self.choose_fitting_shape_combo_box.setCurrentIndex(-1)
            self.choose_fitting_shape_combo_box.setEnabled(False)
            self.shape_name_line_edit.setEnabled(False)
            self.fit_a_shape_label.setEnabled(False)
            self.finish_fitting_a_shape_button.setEnabled(False)
            self.finish_fitting_a_shape_button.hide()
            self.event_type_combo_box.setEnabled(False)
            self.choose_reference_frame_spin_box.setEnabled(False)
            self.event_statistics_x_data_combo_box.setEnabled(False)
            self.event_statistics_y_data_combo_box.setEnabled(False)
            self.plot_event_statistics_botton.setEnabled(False)
            self.event_statistics_window_radius_x_data_label_spin_box.setEnabled(False)
            self.event_statistics_window_radius_y_data_label_spin_box.setEnabled(False)

        if segmentation:
            self.show_segmentation_check_box.setEnabled(True)
            self.save_segmentation_button.setEnabled(True)
            self.export_to_tiff_button.setEnabled(True)
            self.analyze_segmentation_button.setEnabled(True)
            self.analyze_frame_button.setEnabled(True)
            self.fix_segmentation_button.setEnabled(True)
            self.hc_threshold_label.setEnabled(True)
            self.hc_threshold_spin_box.setEnabled(True)
            self.hc_thresholod_percentage_label.setEnabled(True)
            self.peak_window_size_label.setEnabled(True)
            self.hc_threshold_percentage_spin_box.setEnabled(True)
            self.hc_threshold_window_spin_box.setEnabled(True)
            self.cell_size_label_max.setEnabled(True)
            self.cell_size_label_min.setEnabled(True)
            self.cell_size_spin_box_min.setEnabled(True)
            self.cell_size_spin_box_max.setEnabled(True)
            self.fit_a_shape_button.setEnabled(True)
            self.choose_marking_target_combo_box.setEnabled(True)
            self.choose_marking_target_combo_box.clear()
            self.choose_marking_target_combo_box.addItem("Points")
            self.choose_fitting_shape_combo_box.setEnabled(True)
            self.shape_name_line_edit.setEnabled(True)
            self.choose_fitting_shape_combo_box.clear()
            self.choose_fitting_shape_combo_box.addItems(self.tissue_info.FITTING_SHAPES)
            self.fit_a_shape_label.setEnabled(True)
        else:
            self.show_segmentation_check_box.setEnabled(False)
            self.save_segmentation_button.setEnabled(False)
            self.export_to_tiff_button.setEnabled(False)
            self.analyze_frame_button.setEnabled(False)
            self.analyze_segmentation_button.setEnabled(False)
            self.hc_threshold_label.setEnabled(False)
            self.hc_threshold_spin_box.setEnabled(False)
            self.hc_thresholod_percentage_label.setEnabled(False)
            self.peak_window_size_label.setEnabled(False)
            self.hc_threshold_percentage_spin_box.setEnabled(False)
            self.hc_threshold_window_spin_box.setEnabled(False)
            self.cell_size_label_max.setEnabled(False)
            self.cell_size_label_min.setEnabled(False)
            self.cell_size_spin_box_min.setEnabled(False)
            self.cell_size_spin_box_max.setEnabled(False)
            self.fit_a_shape_button.setEnabled(False)
            self.choose_marking_target_combo_box.setEnabled(False)
            self.choose_fitting_shape_combo_box.setEnabled(False)
            self.shape_name_line_edit.setEnabled(False)
            self.fit_a_shape_label.setEnabled(False)
            analysis = False
        if analysis:
            self.show_cell_types_check_box.setEnabled(True)
            self.show_cell_tracking_check_box.setEnabled(True)
            self.cell_tracking_spin_box.setEnabled(True)
            self.show_neighbors_check_box.setEnabled(True)
            self.track_cells_button.setEnabled(True)
            self.find_events_button.setEnabled(True)
            self.plot_single_cell_data_spin_box.setEnabled(True)
            self.plot_single_cell_data_combo_box.setEnabled(True)
            self.plot_single_cell_data_button.setEnabled(True)
            self.plot_event_related_data_push_button.setEnabled(True)
            self.plot_single_frame_data_button.setEnabled(True)
            self.plot_spatial_map_button.setEnabled(True)
            self.spatial_resolution_spin_box.setEnabled(True)
            self.spatial_resolution_label.setEnabled(True)
            self.window_radius_spin_box.setEnabled(True)
            self.window_radius_label.setEnabled(True)
            self.plot_single_frame_data_x_combo_box.setEnabled(True)
            self.plot_single_frame_data_y_combo_box.setEnabled(True)
            self.plot_single_frame_data_cell_type_combo_box.setEnabled(True)
            self.plot_compare_frames_button.setEnabled(True)
            self.compare_frames_line_edit.setEnabled(True)
            self.compare_frames_combo_box.setEnabled(True)
            self.plot_compare_frame_data_cell_type_combo_box.setEnabled(True)
            self.fix_cell_types_button.setEnabled(True)
            self.mark_event_button.setEnabled(True)
            self.delete_events_button.setEnabled(True)
            self.delete_events_frame_check_box.setEnabled(True)
            self.delete_events_aout_detected_check_box.setEnabled(True)
            self.mark_event_combo_box.setEnabled(True)
            self.show_events_check_box.setEnabled(True)
            self.show_events_button.setEnabled(True)
            self.choose_marking_target_combo_box.clear()
            self.choose_marking_target_combo_box.addItems(["Points", "Cells"])
            self.event_type_combo_box.setEnabled(True)
            self.choose_reference_frame_spin_box.setEnabled(True)
            self.event_statistics_x_data_combo_box.setEnabled(True)
            self.event_statistics_y_data_combo_box.setEnabled(True)
            self.plot_event_statistics_botton.setEnabled(True)
            self.event_statistics_window_radius_x_data_label_spin_box.setEnabled(True)
            self.event_statistics_window_radius_y_data_label_spin_box.setEnabled(True)
            self.export_to_tiff_button.setEnabled(True)
        else:
            self.show_cell_types_check_box.setEnabled(False)
            self.show_cell_tracking_check_box.setEnabled(False)
            self.cell_tracking_spin_box.setEnabled(False)
            self.show_neighbors_check_box.setEnabled(False)
            self.track_cells_button.setEnabled(False)
            self.find_events_button.setEnabled(False)
            self.plot_single_cell_data_spin_box.setEnabled(False)
            self.plot_single_cell_data_combo_box.setEnabled(False)
            self.plot_single_cell_data_button.setEnabled(False)
            self.plot_event_related_data_push_button.setEnabled(False)
            self.plot_single_frame_data_button.setEnabled(False)
            self.plot_spatial_map_button.setEnabled(False)
            self.spatial_resolution_spin_box.setEnabled(False)
            self.spatial_resolution_label.setEnabled(False)
            self.window_radius_spin_box.setEnabled(False)
            self.window_radius_label.setEnabled(False)
            self.plot_single_frame_data_x_combo_box.setEnabled(False)
            self.plot_single_frame_data_y_combo_box.setEnabled(False)
            self.plot_single_frame_data_cell_type_combo_box.setEnabled(False)
            self.plot_compare_frames_button.setEnabled(False)
            self.compare_frames_line_edit.setEnabled(False)
            self.compare_frames_combo_box.setEnabled(False)
            self.plot_compare_frame_data_cell_type_combo_box.setEnabled(False)
            self.mark_event_button.setEnabled(False)
            self.delete_events_button.setEnabled(False)
            self.delete_events_frame_check_box.setEnabled(False)
            self.delete_events_aout_detected_check_box.setEnabled(False)
            self.mark_event_combo_box.setEnabled(False)
            self.show_events_check_box.setEnabled(False)
            self.show_events_button.setEnabled(False)
            self.abort_event_marking_button.setEnabled(False)
            self.abort_event_marking_button.hide()
            self.choose_marking_target_combo_box.removeItem(self.choose_marking_target_combo_box.findData("Cells"))
            self.export_to_tiff_button.setEnabled(False)

    def plot_single_cell_data(self):
        if self.plot_single_cell_data_combo_box.currentIndex() < 0:
            return 0
        cell_id = self.plot_single_cell_data_spin_box.value()
        feature = self.plot_single_cell_data_combo_box.currentText()
        plot_window = PlotDataWindow(self, working_dir=self.working_directory)
        if "intensity" in feature:
            intensity_images = self.img[:, self.atoh_spin_box.value(), 0, :, :]
            if not self.img_in_memory:
                intensity_images = intensity_images.compute()
            intensity_images = np.transpose(intensity_images, (0, 2, 1))
        else:
            intensity_images = None
        data = self.tissue_info.plot_single_cell_data(cell_id, feature, plot_window.get_ax(),
                                                      intensity_images=intensity_images,
                                                      window_radius=self.window_radius_spin_box.value())
        plot_window.show(data)

    def plot_event_related_data(self):
        if self.plot_single_cell_data_combo_box.currentIndex() < 0:
            return 0
        frames_around_event = 10
        cell_id = self.plot_single_cell_data_spin_box.value()
        feature = self.plot_single_cell_data_combo_box.currentText()
        frame = self.frame_slider.value()
        plot_window = PlotDataWindow(self, working_dir=self.working_directory)
        if "intensity" in feature:
            frames = np.arange(max(frame - frames_around_event, 0),
                               min(frame + frames_around_event + 1, self.number_of_frames + 1))
            intensity_images = self.img[frames - 1, self.atoh_spin_box.value(), 0, :, :]
            if not self.img_in_memory:
                intensity_images = intensity_images.compute()
            intensity_images = np.transpose(intensity_images, (0, 2, 1))
        else:
            intensity_images = None
        data = self.tissue_info.plot_event_related_data(cell_id, frame, feature, frames_around_event,
                                                        plot_window.get_ax(), intensity_images=intensity_images)
        plot_window.show(data)

    def plot_single_frame_data(self):
        if self.plot_single_frame_data_x_combo_box.currentIndex() < 0 or \
                self.plot_single_frame_data_y_combo_box.currentIndex() < 0 or \
                self.plot_single_frame_data_cell_type_combo_box.currentIndex() < 0:
            return 0
        x_feature = self.plot_single_frame_data_x_combo_box.currentText()
        y_feature = self.plot_single_frame_data_y_combo_box.currentText()
        cell_type = self.plot_single_frame_data_cell_type_combo_box.currentText()
        frame = self.frame_slider.value()
        if "intensity" in x_feature or "intensity" in y_feature:
            intensity_image = self.img[frame - 1, self.atoh_spin_box.value(), 0, :, :].T
            if not self.img_in_memory:
                intensity_image = intensity_image.compute()
        else:
            intensity_image = None
        plot_window = PlotDataWindow(self, working_dir=self.working_directory)
        data, error_message = self.tissue_info.plot_single_frame_data(frame, x_feature, y_feature, plot_window.get_ax(),
                                                                      cell_type, intensity_image=intensity_image)
        if error_message:
            message_box = QtWidgets.QMessageBox
            message_box.question(self, '', error_message, message_box.Close)
        else:
            plot_window.show(data)

    def plot_spatial_map(self):
        if self.compare_frames_combo_box.currentIndex() < 0 or \
                self.plot_compare_frame_data_cell_type_combo_box.currentIndex() < 0:
            return 0
        feature = self.compare_frames_combo_box.currentText()
        cell_type = self.plot_compare_frame_data_cell_type_combo_box.currentText()
        window_radius = self.window_radius_spin_box.value()
        spatial_resolution = self.spatial_resolution_spin_box.value()
        frame = self.frame_slider.value()
        plot_window = PlotDataWindow(self, working_dir=self.working_directory)
        data, error_message = self.tissue_info.plot_spatial_map(frame, feature, window_radius, spatial_resolution,
                                                                plot_window.get_ax(), cells_type=cell_type)
        if error_message:
            message_box = QtWidgets.QMessageBox
            message_box.question(self, '', error_message, message_box.Close)
        else:
            plot_window.show(data)

    def display_events(self):
        events = self.tissue_info.get_events()
        model = PandasModel(events)
        window = QtWidgets.QMainWindow(self)
        window.setWindowTitle("Events table")
        view = QtWidgets.QTableView(window)
        view.setModel(model)
        view.resize(1500, 600)
        window.resize(1500, 600)
        window.show()

    def plot_compare_frames_data(self):
        if self.compare_frames_combo_box.currentIndex() < 0:
            return 0
        frames_string = self.compare_frames_line_edit.text()
        split_frame_string =  re.findall("[-\d]+", frames_string)
        frames = []
        for frame_string in split_frame_string:
            frame_string = re.sub(r"\s+", "", frame_string)
            if re.match('\d-\d', frame_string):
                boundaries = frame_string.split("-")
                frames.extend(np.arange(start=int(boundaries[0]), stop=int(boundaries[1])+1))
            elif re.match('\d', frame_string):
                frames.append(int(frame_string))
        feature = self.compare_frames_combo_box.currentText()
        cell_type = self.plot_compare_frame_data_cell_type_combo_box.currentText()
        plot_window = PlotDataWindow(self, working_dir=self.working_directory)
        data, error_message = self.tissue_info.plot_compare_frames_data(frames, feature, plot_window.get_ax(),
                                                                        cell_type)
        if error_message:
            message_box = QtWidgets.QMessageBox
            message_box.question(self, '', error_message, message_box.Close)
        else:
            plot_window.show(data)

    def plot_event_statistics(self):
        if self.event_type_combo_box.currentIndex() < 0:
            return 0
        else:
            event_type = self.event_type_combo_box.currentText()
        if self.event_statistics_x_data_combo_box.currentIndex() < 0:
            return 0
        else:
            x_feature = self.event_statistics_x_data_combo_box.currentText()
            x_radius = self.event_statistics_window_radius_x_data_label_spin_box.value()
        if self.event_statistics_y_data_combo_box.currentIndex() < 0:
            y_feature = None
            y_radius = None
        else:
            y_feature = self.event_statistics_y_data_combo_box.currentText()
            if y_feature == "None":
                y_feature = None
            y_radius = self.event_statistics_window_radius_y_data_label_spin_box.value()
        plot_window = PlotDataWindow(self, working_dir=self.working_directory)
        if "reference" in event_type:
            cells_type, positive_for_type = ("HC", True) if "HC" in event_type else ("HC", False) if "SC" in event_type else ("all", True)
            reference_frame = self.choose_reference_frame_spin_box.value()
            if "intensity" in x_feature or (y_feature is not None and "intensity" in y_feature):
                intensity_image = self.img[reference_frame - 1, self.atoh_spin_box.value(), 0, :, :]
                if not self.img_in_memory:
                    intensity_image = intensity_image.compute()
                intensity_image = intensity_image.T
            else:
                intensity_image = None
            data, error_message = self.tissue_info.plot_overall_statistics(reference_frame, x_feature, y_feature,
                                                                           plot_window.get_ax(),
                                                                           intensity_img=intensity_image,
                                                                           x_cells_type=cells_type,
                                                                           y_cells_type=cells_type,
                                                                           x_radius=x_radius, y_radius=y_radius)
        else:
            if "intensity" in x_feature or (y_feature is not None and "intensity" in y_feature):
                intensity_images = self.img[:, self.atoh_spin_box.value(), 0, :, :]
                if not self.img_in_memory:
                    intensity_images = intensity_images.compute()
                intensity_images = np.transpose(intensity_images, (0, 2, 1))
            else:
                intensity_images = None
            data, error_message = self.tissue_info.plot_event_statistics(event_type, x_feature, x_radius, y_feature,
                                                                         y_radius, plot_window.get_ax(),
                                                                         intensity_images=intensity_images)
        if error_message:
            message_box = QtWidgets.QMessageBox
            message_box.question(self, '', error_message, message_box.Close)
        if data is not None:
            plot_window.show(data)

    def frame_segmentation_and_analysis_done(self, msg):
        self.frame_segmentation_done(msg)
        self.frame_analysis_done(msg)

    def frame_segmentation_done(self, msg):
        split_msg = msg.split("/")
        percentage_done = int(split_msg[1])
        self.segmentation_saved = False
        self.segment_all_frames_progress_bar.setValue(percentage_done)
        if percentage_done == 100:
            frame = self.frame_slider.value()
            self.current_segmentation = self.tissue_info.get_segmentation(frame)
            self.segmentation_changed = True
            self.setState(image=True, segmentation=True, analysis=False)
            self.display_frame()
            self.frame_slider.setEnabled(True)
            self.frame_line_edit.setEnabled(True)
            self.segment_all_frames_button.setEnabled(True)
            self.cancel_segmentation_button.hide()
            self.segment_all_frames_progress_bar.hide()
            if self.epyseg_dir is not None:
                shutil.rmtree(self.epyseg_dir)
                self.epyseg_dir = None

    def segment_frames(self, frame_numbers):
        self.segment_all_frames_progress_bar.reset()
        self.segment_all_frames_progress_bar.show()
        self.cancel_segmentation_button.show()
        zo_channel = self.zo_spin_box.value()
        threshold = 0.01 * self.segmentation_threshold_spin_box.value()
        std = self.segmentation_kernel_std_spin_box.value()
        block_size = self.segmentation_block_size_spin_box.value()
        self.segmentation_thread = SegmentAllThread(self.img, zo_channel, threshold, std, block_size,
                                                    frame_numbers,self.img_in_memory)
        self.segmentation_thread._signal.connect(self.frame_segmentation_and_analysis_done)
        self.segment_all_frames_button.setEnabled(False)
        self.frame_slider.setEnabled(False)
        self.frame_line_edit.setEnabled(False)
        self.segmentation_thread.start()

    def segment_frame(self, frame_number=0):
        if not self.data_lost_warning(self.segment_frame):
            return 0
        if frame_number == 0:
            frame_number = self.frame_slider.value()
        else:
            self.frame_slider.setValue(frame_number)
        if self.segment_using_Unet_question():
            self.segment_frames_using_ShachafNET([frame_number])
        elif self.segment_using_epyseg_question():
            self.segment_frames_using_epyseg([frame_number])
        else:
            self.segment_frames([frame_number])

    def segment_frames_using_epyseg(self, frames):
        self.segment_all_frames_progress_bar.reset()
        self.segment_all_frames_progress_bar.show()
        self.cancel_segmentation_button.show()
        self.epyseg_dir = os.path.join(self.working_directory, "epyseg_temp_folder")
        if not os.path.exists(self.epyseg_dir):
            os.mkdir(self.epyseg_dir)
        zo_channel = self.zo_spin_box.value()
        self.segmentation_thread = SaveImagesThread(self.img, self.epyseg_dir, frames, zo_channel, self.img_in_memory)
        self.segmentation_thread._signal.connect(self.saving_segmentation_images_done)
        self.segment_all_frames_button.setEnabled(False)
        self.segmentation_thread.start()

    def segment_frames_using_ShachafNET(self, frames):
        self.segment_all_frames_progress_bar.reset()
        self.segment_all_frames_progress_bar.show()
        self.cancel_segmentation_button.show()
        self.analyze_segmentation_progress_bar.reset()
        self.analyze_segmentation_progress_bar.show()
        self.cancel_analysis_button.show()
        zo_channel = self.zo_spin_box.value()
        atoh_channel = self.atoh_spin_box.value()
        self.segmentation_thread = UnetSegmentationThread(self.img, frames, zo_channel, atoh_channel, self.tissue_info,
                                                          self.img_in_memory)
        self.segmentation_thread._signal.connect(self.frame_segmentation_and_analysis_done)
        self.segment_all_frames_button.setEnabled(False)
        self.segmentation_thread.start()

    def saving_segmentation_images_done(self, msg):
        percentage_done = int(msg)
        self.segmentation_saved = False
        self.segment_all_frames_progress_bar.setValue(percentage_done)
        if percentage_done == 100:
            frames = self.segmentation_thread.frames
            self.segment_all_frames_button.setEnabled(True)
            self.cancel_segmentation_button.hide()
            self.segment_all_frames_progress_bar.hide()
            message_box = QtWidgets.QMessageBox
            message_box.about(self, '', "Loading EPySeg. Use predict on input folder %s" % self.epyseg_dir)
            sys.path.insert(1, '..\\EPySeg')
            from epyseg.epygui import EPySeg
            self.epyseg = EPySeg()
            self.epyseg.show()
            self.segment_all_frames_progress_bar.reset()
            self.segment_all_frames_progress_bar.show()
            self.cancel_segmentation_button.show()
            self.segmentation_thread = ExternalSegmentationThread(self.tissue_info, self.epyseg_dir, frames)
            self.segmentation_thread._signal.connect(self.frame_segmentation_done)
            self.segment_all_frames_button.setEnabled(False)
            self.segmentation_thread.start()

    def this_might_take_a_while_message(self):
        message_box = QtWidgets.QMessageBox
        ret = message_box.question(self, '', "Are you sure? this might take a while...",
                                   message_box.Yes | message_box.No)
        return ret == message_box.Yes

    def segment_all_frames(self):
        if not self.data_lost_warning(self.segment_all_frames):
            return 0
        if self.segment_using_Unet_question():
            self.segment_frames_using_ShachafNET(np.arange(1,self.number_of_frames+1))
        elif self.segment_using_epyseg_question():
            self.segment_frames_using_epyseg(np.arange(1,self.number_of_frames+1))
        else:
            if self.this_might_take_a_while_message():
                self.segment_frames(np.arange(1,self.number_of_frames+1))

    def cancel_segmentation(self):
        self.segmentation_thread.kill()

    def segment_using_epyseg_question(self):
        message_box = QtWidgets.QMessageBox
        ret = message_box.question(self, '',
                                   "Do you want to use epySeg (Deep-Learning based segmentation, will open up in a new window)?",
                                   message_box.Yes | message_box.No)
        return ret == message_box.Yes

    def segment_using_Unet_question(self):
        message_box = QtWidgets.QMessageBox
        ret = message_box.question(self, '',
                                   "Do you want to use Unet (Deep-Learning based segmentation)?",
                                   message_box.Yes | message_box.No)
        return ret == message_box.Yes

    def frame_analysis_done(self, msg):
        split_msg = msg.split("/")
        percentage_done = int(split_msg[1])
        self.analysis_saved = False
        self.analyze_segmentation_progress_bar.setValue(percentage_done)

        if percentage_done == 100:
            self.cancel_analysis_button.hide()
            self.analyze_segmentation_progress_bar.hide()
            self.setState(image=True, segmentation=True, analysis=True)
            self.cells_number_changed()
            self.update_single_cell_features()
            self.update_single_frame_features()
            self.frame_slider.setEnabled(True)
            self.frame_line_edit.setEnabled(True)
            self.analysis_changed = True
            self.display_frame()

    def find_cell_types_in_frames(self, frame_numbers):
        self.analyze_segmentation_progress_bar.reset()
        self.analyze_segmentation_progress_bar.show()
        self.cancel_analysis_button.show()
        atoh_channel = self.atoh_spin_box.value()
        hc_threshold = self.hc_threshold_spin_box.value()/100
        hc_threshold_percentage = self.hc_threshold_percentage_spin_box.value()
        if self.peak_window_size_label.isChecked():
            peak_window_radius = self.hc_threshold_window_spin_box.value()
        else:
            peak_window_radius = 0
        type_name = self.channel_names[self.atoh_spin_box.value()]
        self.analysis_thread = CellTypesThread(self.img, self.tissue_info, frame_numbers, atoh_channel, hc_threshold,
                                              hc_threshold_percentage, peak_window_radius,
                                              type_name, self.img_in_memory)
        self.analysis_thread._signal.connect(self.frame_analysis_done)
        self.analyze_segmentation_button.setEnabled(False)
        self.frame_slider.setEnabled(False)
        self.frame_line_edit.setEnabled(False)
        self.analysis_thread.start()

    def find_cell_types_in_frame(self, frame_number=0):
        if not self.data_lost_warning(self.find_cell_types_in_frame):
            return 0
        if frame_number == 0:
            frame_number = self.frame_slider.value()
        else:
            self.frame_slider.setValue(frame_number)
        self.find_cell_types_in_frames([frame_number])

    def analyze_segmentation(self):
        if not self.data_lost_warning(self.analyze_segmentation):
            return 0
        if self.this_might_take_a_while_message():
            self.find_cell_types_in_frames(np.arange(1, self.number_of_frames+1))

    def cancel_analysis(self):
        self.analysis_thread.kill()

    def get_analysis_img(self, types, neighbors, track, track_cell_label=0, events=False, marking_points=False):
        frame_number = self.frame_slider.value()
        if types or neighbors or track or events or marking_points:
            img = np.zeros(self.current_frame.shape)
            if types:
                img += self.tissue_info.draw_cell_types(frame_number, type_name=self.channel_names[self.atoh_spin_box.value()])
            if neighbors:
                img += self.tissue_info.draw_neighbors_connections(frame_number)
            if track:
                img += self.tissue_info.draw_cell_tracking(frame_number, track_cell_label)
            if events:
                img += self.tissue_info.draw_events(frame_number)
            if marking_points:
                img += self.tissue_info.draw_marking_points(frame_number)
            return np.clip(img, 0, 1)
        else:
            return None

    def cells_tracking_done(self, msg):
        percentage_done = int(msg)
        self.analysis_saved = False
        self.track_cells_progress_bar.setValue(percentage_done)

        if percentage_done == 100:
            self.cells_number_changed()
            self.cancel_tracking_button.hide()
            self.track_cells_progress_bar.hide()
            self.setState(image=True, segmentation=True, analysis=True)
            self.frame_slider.setEnabled(True)
            self.frame_line_edit.setEnabled(True)
            self.tissue_info.fix_cell_id_in_events()

    def event_finding_done(self, msg):
        percentage_done = int(msg)
        self.analysis_saved = False
        self.find_events_progress_bar.setValue(percentage_done)

        if percentage_done == 100:
            self.cancel_event_finding_button.hide()
            self.find_events_progress_bar.hide()
            self.setState(image=True, segmentation=True, analysis=True)
            self.frame_slider.setEnabled(True)
            self.frame_line_edit.setEnabled(True)
            self.show_events_button.show()
            self.show_events_button.setEnabled(True)
            self.find_events_button.setEnabled(True)
            self.find_events_button.show()

    def correct_tracking(self, off=False):
        if off:
            self.fix_tracking_label.hide()
            self.fix_tracking_button.show()
            self.show_cell_tracking_check_box.show()
            self.cell_tracking_spin_box.show()
            self.fix_tracking_button.show()
            self.fix_tracking_on = False
        else:
            self.fix_tracking_button.hide()
            self.show_cell_tracking_check_box.hide()
            self.cell_tracking_spin_box.hide()
            self.fix_tracking_button.hide()
            self.fix_tracking_label.show()
            self.fix_tracking_on = True

    def track_cells(self, initial_frame=-1, final_frame=-1):
        self.track_cells_progress_bar.reset()
        self.track_cells_progress_bar.show()
        if (initial_frame == -1 or not initial_frame) and final_frame == -1:
            initial_frame = 1
            final_frame = self.number_of_frames
        self.cancel_tracking_button.show()
        self.tracking_thread = TrackingThread(self.tissue_info, self.img[:, self.zo_spin_box.value(), 0, :, :], self.img_in_memory,
                                              initial_frame=initial_frame, final_frame=final_frame)
        self.tracking_thread._signal.connect(self.cells_tracking_done)
        self.track_cells_button.setEnabled(False)
        self.frame_slider.setEnabled(False)
        self.frame_line_edit.setEnabled(False)
        self.tracking_thread.start()

    def cancel_tracking(self):
        self.tracking_thread.kill()

    def find_events(self, initial_frame=-1, final_frame=-1):
        self.find_events_progress_bar.reset()
        self.find_events_progress_bar.show()
        if (initial_frame == -1 or not initial_frame) and final_frame == -1:
            initial_frame = 1
            final_frame = self.number_of_frames
        self.cancel_event_finding_button.show()
        self.event_finding_thread = EventFindingThread(self.tissue_info, initial_frame, final_frame)
        self.event_finding_thread._signal.connect(self.event_finding_done)
        self.find_events_button.setEnabled(False)
        self.find_events_button.hide()
        self.frame_slider.setEnabled(False)
        self.frame_line_edit.setEnabled(False)
        self.show_events_button.setEnabled(False)
        self.show_events_button.hide()
        self.event_finding_thread.start()

    def cancel_event_finding(self):
        self.event_finding_thread.kill()

    def fix_segmentation(self):
        self.fix_segmentation_button.setEnabled(False)
        self.fix_segmentation_button.hide()
        self.fix_segmentation_label.setEnabled(True)
        self.fix_segmentation_label.show()
        self.finish_fixing_segmentation_button.setEnabled(True)
        self.finish_fixing_segmentation_button.show()
        self.frame_slider.setEnabled(False)
        self.frame_line_edit.setEnabled(False)
        self.fixing_segmentation_mode = FIX_SEGMENTATION_ON
        self.fix_cell_types_button.setEnabled(False)


    def finish_fixing_segmentation(self):
        frame = self.frame_slider.value()
        if self.fix_segmentation_last_position is not None:
            self.tissue_info.add_segmentation_line(frame, self.fix_segmentation_last_position, final=True)
            self.fix_segmentation_last_position = None
        self.tissue_info.update_labels(self.frame_slider.value())
        self.segmentation_changed = True
        self.current_segmentation = self.tissue_info.get_segmentation(self.frame_slider.value())
        self.display_frame()
        self.fixing_segmentation_mode = FIX_SEGMENTATION_OFF
        self.cells_number_changed()
        self.fix_segmentation_button.setEnabled(True)
        self.fix_segmentation_button.show()
        self.fix_segmentation_label.setEnabled(False)
        self.fix_segmentation_label.hide()
        self.finish_fixing_segmentation_button.setEnabled(False)
        self.finish_fixing_segmentation_button.hide()
        self.frame_slider.setEnabled(True)
        self.frame_line_edit.setEnabled(True)
        self.fix_cell_types_button.setEnabled(True)

    def fix_cell_types(self):
        self.fix_cell_types_button.setEnabled(False)
        self.fix_cell_types_button.hide()
        self.fix_cell_types_label.setEnabled(True)
        self.fix_cell_types_label.show()
        self.finish_fixing_cell_types_button.setEnabled(True)
        self.finish_fixing_cell_types_button.show()
        self.remove_non_sensory_button.setEnabled(True)
        self.remove_non_sensory_button.show()
        self.frame_slider.setEnabled(False)
        self.frame_line_edit.setEnabled(False)
        self.fix_cell_types_on = True
        self.fix_segmentation_button.setEnabled(False)
        self.analysis_changed = True

    def finish_fixing_cell_types(self):
        self.fix_cell_types_on = False
        self.fix_cell_types_button.setEnabled(True)
        self.fix_cell_types_button.show()
        self.fix_cell_types_label.setEnabled(False)
        self.fix_cell_types_label.hide()
        self.finish_fixing_cell_types_button.setEnabled(False)
        self.finish_fixing_cell_types_button.hide()
        self.remove_non_sensory_button.setEnabled(False)
        self.remove_non_sensory_button.hide()
        self.frame_slider.setEnabled(True)
        self.frame_line_edit.setEnabled(True)
        self.fix_segmentation_button.setEnabled(True)

    def remove_nonsensory_cells(self):
        self.tissue_info.remove_cells_outside_of_sensory_region(self.frame_slider.value())
        self.analysis_changed = True
        self.display_frame()

    def fit_a_shape(self, pos=None):
        if self.fitting_stage == 1:
            self.tissue_info.add_shape_fitting_point(self.frame_slider.value(),
                                                     pos, self.choose_marking_target_combo_box.currentText())
            self.analysis_changed = True
            self.display_frame()
        else:
            self.tissue_info.start_shape_fitting()
            self.fitting_stage = 1
            self.finish_fitting_a_shape_button.setEnabled(True)
            self.choose_marking_target_combo_box.setEnabled(False)
            self.choose_fitting_shape_combo_box.setEnabled(False)
            self.shape_name_line_edit.setEnabled(False)
            self.finish_fitting_a_shape_button.show()
            message_box = QtWidgets.QMessageBox
            message_box.about(self, '', 'Mark desired %s. When finished click on \"Finish marking\"' %
                              self.choose_marking_target_combo_box.currentText())

    def finish_fitting_a_shape(self):
        self.fitting_stage = 0
        plot_window = PlotDataWindow(self, working_dir=self.working_directory)
        shape_name = self.shape_name_line_edit.text()
        res = self.tissue_info.end_shape_fitting(self.frame_slider.value(),
                                                 self.choose_fitting_shape_combo_box.currentText(),
                                                 plot_window.get_ax(), shape_name)
        new_features = ["%s:%s" % (shape_name, key) for key in res.keys()]
        for feature in new_features:
            if self.compare_frames_combo_box.findText(feature) == -1:
                self.compare_frames_combo_box.addItem(feature)
        self.finish_fitting_a_shape_button.setEnabled(False)
        self.finish_fitting_a_shape_button.hide()
        self.choose_marking_target_combo_box.setEnabled(True)
        self.choose_fitting_shape_combo_box.setEnabled(True)
        self.shape_name_line_edit.setEnabled(True)
        print(res)
        plot_window.show()


    def save_data(self):
        name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', directory=self.working_directory)[0]
        if name:
            self.save_data_progress_bar.reset()
            self.save_data_progress_bar.show()
            self.frame_slider.setEnabled(False)
            self.frame_line_edit.setEnabled(False)
            self.save_thread = SaveDataThread(self.tissue_info, name)
            self.save_thread._signal.connect(self.frame_saving_done)
            self.save_thread.start()
        return 0

    def export_segmentation_to_tiff(self):
        f_name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', directory=self.working_directory)[0]
        if f_name:
            filename = os.path.basename(f_name)
            outfolder = os.path.dirname(f_name)
            self.tissue_info.export_segmentation_to_tiff(outfolder, filename, arr_shape=(self.img_dimensions.Y, self.img_dimensions.X))

    def frame_saving_done(self, msg):
        percentage_done = msg
        self.save_data_progress_bar.setValue(percentage_done)
        if percentage_done == 100:
            self.segmentation_saved = True
            self.analysis_saved = True
            self.save_data_progress_bar.hide()
            self.frame_slider.setEnabled(True)
            self.frame_line_edit.setEnabled(True)
            for func in self.waiting_for_data_save:
                func()
            self.waiting_for_data_save = []

    def data_lost_warning(self, calling_function):
        message_box = QtWidgets.QMessageBox
        if not self.segmentation_saved or not self.analysis_saved:
            ret = message_box.question(self, '', "Current data will be lost, do you want to save it first?",
                                       message_box.Save | message_box.Discard | message_box.Cancel)
            if ret == message_box.Cancel:
                return False
            elif ret == message_box.Save:
                self.waiting_for_data_save.append(calling_function)
                self.save_data()
                return False
        return True

    def load_data(self):
        if not self.data_lost_warning(self.load_data):
            return 0
        name = QtWidgets.QFileDialog.getOpenFileName(caption='Open File',
                                            directory=self.working_directory, filter="*.seg")[0]
        if name:
            self.load_data_progress_bar.reset()
            self.load_data_progress_bar.show()
            self.load_thread = LoadDataThread(self.tissue_info, name, self.channel_names[self.atoh_spin_box.value()])
            self.load_thread._signal.connect(self.frame_loading_done)
            try:
                self.load_thread.start()
            except shutil.ReadError:
                message_box = QtWidgets.QMessageBox
                message_box.about(self, '', 'Could not load file %s' % name[0])
        return 0

    def frame_loading_done(self, msg):
        percentage_done = msg
        self.load_data_progress_bar.setValue(percentage_done)
        if percentage_done == 100:
            self.current_segmentation = self.tissue_info.get_segmentation(self.frame_slider.value())
            self.segmentation_changed = True
            self.display_frame()
            self.load_data_progress_bar.hide()
            self.setState(image=True, segmentation=self.tissue_info.is_any_segmented(),
                          analysis=self.tissue_info.is_any_analyzed(type_name=self.channel_names[self.atoh_spin_box.value()]))
            channel_names = self.tissue_info.get_channel_names()
            if channel_names:
                self.channel_names = channel_names
            fake_channels = self.tissue_info.get_fake_channels()
            if fake_channels:
                self.fake_channels = fake_channels
            self.atoh_spin_box.setMaximum(len(self.channel_names))
            self.zo_spin_box.setMaximum(len(self.channel_names))
            self.cells_number_changed()
            self.update_single_cell_features()
            self.update_single_frame_features()
            self.update_shape_fitting()



class SegmentAllThread(QtCore.QThread):
    _signal = QtCore.pyqtSignal(str)

    def __init__(self, img, zo_channel, threshold, std, block_size, frame_numbers, img_in_memory=False):
        super(SegmentAllThread, self).__init__()
        self.img = img
        self.zo_channel = zo_channel
        self.frame_numbers = frame_numbers
        self.threshold = threshold
        self.std = std
        self.block_size = block_size
        self.is_killed = False
        self.img_in_memory = img_in_memory

    def __del__(self):
        self.wait()

    def run(self):
        done_frames = 0
        for frame in self.frame_numbers:
            if self.img_in_memory:
                zo_img = self.img[frame - 1, self.zo_channel, 0, :, :].T
            else:
                zo_img = self.img[frame - 1, self.zo_channel, 0, :, :].compute().T
            labels = watershed_segmentation(zo_img, self.threshold, self.std, self.block_size)
            self.tissue_info.set_labels(frame, labels, reset_data=True)
            self.tissue_info.calculate_frame_cellinfo(frame)
            done_frames += 1
            percentage_done = np.round(100*done_frames/len(self.frame_numbers))
            self.emit("%d/%d" % (frame, percentage_done))
            if self.is_killed:
                self.emit("%d/%d" % (frame, 100))
                break

    def emit(self, msg):
        self._signal.emit(msg)

    def kill(self):
        self.is_killed = True


class CellTypesThread(QtCore.QThread):
    _signal = QtCore.pyqtSignal(str)

    def __init__(self, img, tissue_info, frame_numbers, atoh_channel, hc_threshold,
                 percentage_above_threshold, peak_window_radius,
                 type_name="HC",
                 img_in_memory=False):
        super(CellTypesThread, self).__init__()
        self.tissue_info = tissue_info
        self.img = img
        self.frame_numbers = frame_numbers
        self.atoh_channel = atoh_channel
        self.hc_threshold = hc_threshold
        self.percentage_above_threshold = percentage_above_threshold
        self.peak_window_radius = peak_window_radius
        self.is_killed = False
        self.type_name = type_name
        self.img_in_memory = img_in_memory

    def __del__(self):
        self.wait()

    def run(self):
        done_frames = 0
        for frame in self.frame_numbers:
            if self.img_in_memory:
                atoh_img = self.img[frame - 1, self.atoh_channel, 0, :, :].T
            else:
                atoh_img = self.img[frame - 1, self.atoh_channel, 0, :, :].compute().T
            self.tissue_info.calc_cell_types(atoh_img, frame, self.type_name, threshold=self.hc_threshold,
                            percentage_above_threshold=self.percentage_above_threshold, peak_window_size=self.peak_window_radius)
            done_frames += 1
            percentage_done = np.round(100*done_frames/len(self.frame_numbers))
            self.emit("%d/%d" % (frame, percentage_done))
            if self.is_killed:
                self.emit("%d/%d" % (frame, 100))
                break

    def emit(self, msg):
        self._signal.emit(msg)

    def kill(self):
        self.is_killed = True


class TrackingThread(QtCore.QThread):
    _signal = QtCore.pyqtSignal(str)

    def __init__(self, tissue_info, images, img_in_memory=False, initial_frame=-1, final_frame=-1):
        super(TrackingThread, self).__init__()
        self.tissue_info = tissue_info
        self.images = images
        self.is_killed = False
        self.img_in_memory = img_in_memory
        self.initial_frame = initial_frame
        self.final_frame = final_frame

    def __del__(self):
        self.wait()

    def run(self):
        tracking_generator = self.tissue_info.track_cells_iterator_with_trackpy(initial_frame=self.initial_frame,
                                                                   final_frame=self.final_frame, images=self.images,
                                                                                image_in_memory=self.img_in_memory)
        for frame in tracking_generator:
            percentage_done = np.round(100 * frame / self.tissue_info.number_of_frames)
            self.emit("%d" % percentage_done)
            if self.is_killed:
                break
        self.emit("%d" % 100)

    def emit(self, msg):
        self._signal.emit(msg)

    def kill(self):
        self.is_killed = True

class EventFindingThread(QtCore.QThread):
    _signal = QtCore.pyqtSignal(str)

    def __init__(self, tissue_info, initial_frame=-1, final_frame=-1):
        super(EventFindingThread, self).__init__()
        self.tissue_info = tissue_info
        self.is_killed = False
        self.initial_frame = initial_frame
        self.final_frame = final_frame

    def __del__(self):
        self.wait()

    def run(self):
        event_finding_generator = self.tissue_info.find_events_iterator(initial_frame=self.initial_frame,
                                                                        final_frame=self.final_frame)
        for frame in event_finding_generator:
            percentage_done = np.round(100 * frame / self.tissue_info.number_of_frames)
            self.emit("%d" % percentage_done)
            if self.is_killed:
                break
        self.emit("%d" % 100)

    def emit(self, msg):
        self._signal.emit(msg)

    def kill(self):
        self.is_killed = True

class SaveDataThread(QtCore.QThread):
    _signal = QtCore.pyqtSignal(int)

    def __init__(self, tissue_info, file_path):
        super(SaveDataThread, self).__init__()
        self.tissue_info = tissue_info
        self.path = file_path

    def __del__(self):
        self.wait()

    def run(self):
        for percent_done in self.tissue_info.save(self.path):
            self.emit(percent_done)
        self.emit(100)

    def emit(self, msg):
        self._signal.emit(int(msg))


class LoadDataThread(QtCore.QThread):
    _signal = QtCore.pyqtSignal(int)

    def __init__(self, tissue_info, file_path, type_name):
        super(LoadDataThread, self).__init__()
        self.tissue_info = tissue_info
        self.path = file_path
        self.type_name = type_name

    def __del__(self):
        self.wait()

    def run(self):
        for percent_done in self.tissue_info.load(self.path, type_name=self.type_name):
            self.emit(percent_done)
        self.emit(100)

    def emit(self, msg):
        self._signal.emit(int(msg))

class SaveImagesThread(QtCore.QThread):
    _signal = QtCore.pyqtSignal(int)

    def __init__(self, img, output_folder, frames, channel, img_in_memory=False):
        super(SaveImagesThread, self).__init__()
        self.img = img
        self.output = output_folder
        self.frames = frames
        self.frames_done = 0
        self.kill = False
        self.channel = channel
        self.img_in_memory = img_in_memory

    def __del__(self):
        self.wait()

    def run(self):
        for frame in self.frames:
            save_path = os.path.join(self.output, "frame_%d_zo.tif"%frame)
            if not os.path.isfile(save_path):
                if self.img_in_memory:
                    frame_img = self.img[frame - 1, self.channel, 0, :, :].T
                else:
                    frame_img = self.img[frame - 1, self.channel, 0, :, :].compute().T
                save_tiff(save_path, frame_img, axes="YX", data_type="uint16")
            self.frames_done += 1
            self.emit(100*self.frames_done/len(self.frames))

    def emit(self, msg):
        self._signal.emit(int(msg))

class UnetSegmentationThread(QtCore.QThread):
    _signal = QtCore.pyqtSignal(str)

    def __init__(self, img, frame_numbers, zo_channel, atoh_channel, out, img_in_memory=False):
        super(UnetSegmentationThread, self).__init__()
        self.img = img
        self.zo_channel = zo_channel
        self.atoh_channel = atoh_channel
        self.out = out
        self.frame_numbers = frame_numbers
        self.is_killed = False
        self.img_in_memory = img_in_memory

    def __del__(self):
        self.wait()

    def run(self):
        done_frames = 0
        for frame in self.frame_numbers:
            if self.img_in_memory:
                img = self.img[frame - 1, [self.atoh_channel, self.zo_channel], 0, :, :]
            else:
                img = self.img[frame - 1, [self.atoh_channel, self.zo_channel], 0, :, :].compute()
            predictor = SegmentationPredictor(UNET_WEIGHTS_PATH, img.shape)
            labels, HC = predictor.predict(img)
            self.out.set_labels(frame, labels, reset_data=True)
            self.out.calculate_frame_cellinfo(frame, hc_marker_image=HC, hc_threshold=0.5,
                                     use_existing_types=False,
                                     percentage_above_HC_threshold=50, peak_window_radius=10)
            done_frames += 1
            percentage_done = np.round(100*done_frames/len(self.frame_numbers))
            self.emit("%d/%d" % (frame, percentage_done))
            if self.is_killed:
                self.emit("%d/%d" % (frame, 100))
                break

    def emit(self, msg):
        self._signal.emit(msg)

    def kill(self):
        self.is_killed = True

class ExternalSegmentationThread(QtCore.QThread):
    _signal = QtCore.pyqtSignal(str)

    def __init__(self, tissue_info, output_folder, frames):
        super(ExternalSegmentationThread, self).__init__()
        self.tissue_info = tissue_info
        self.output = output_folder
        self.frames = frames
        self.frames_done = 0
        self.kill = False
        self.event_handler = PatternMatchingEventHandler(['*'], None, True, True)
        self.event_handler.on_created = self.on_created
        self.observer = Observer()
        self.observer.schedule(self.event_handler, self.output, recursive=True)

    def on_created(self, event):
        path = event.src_path
        old_size = -1
        file_size = os.path.getsize(path)
        while old_size != file_size:
            old_size = file_size
            time.sleep(1)
            file_size = os.path.getsize(path)
        self.load_file(path)

    def __del__(self):
        self.observer.stop()
        self.observer.join()
        self.wait()

    def load_file(self, path):
        file_name = os.path.basename(path)
        if file_name.startswith("frame"):
            frame_number = int(file_name.split("_")[1])
            self.tissue_info.load_labels_from_external_file(frame_number, path)
            self.tissue_info.calculate_frame_cellinfo(frame_number)
            self.frames_done += 1
            self.emit("%d/%d" %(frame_number,100*self.frames_done/len(self.frames)))

    def load_existing_files(self):
        path = os.path.join(self.output, "predict")
        if os.path.isdir(path):
            for filename in os.listdir(path):
                f = os.path.join(path, filename)
                if os.path.isfile(f):
                    self.load_file(f)

    def run(self):
        self.load_existing_files()
        self.observer.start()
        while not self.kill:
            time.sleep(1)


    def emit(self, msg):
        self._signal.emit(msg)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    w = FormImageProcessing()
    w.show()
    sys.exit(app.exec_())