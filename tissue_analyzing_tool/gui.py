import cv2
import numpy as np
import os.path, shutil
import re
from basic_image_manipulations import *
import vispy
import pickle
from tissue_info import Tissue
vispy.use('PyQt5')
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox, QProgressBar, QShortcut
from PyQt5.QtGui import QIcon, QPixmap, QImage, qRgb, QKeySequence
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from functools import partial
COLORTABLE=[]
fname = ""
from numexpr import utils

utils.MAX_THREADS = 8

FIX_SEGMENTATION_OFF = 0
FIX_SEGMENTATION_ON = 1
FIX_SEGMENTATION_LINE = 2

class FormImageProcessing(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        loadUi("movie_display.ui", self)
        self.setWindowTitle("Movie Segmentation")
        self.setState()
        self.saveFile = QShortcut(QKeySequence("Ctrl+S"), self)
        self.undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.connect_methods()
        self.hide_progress_bars()
        self.img = None
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
        self.analysis_saved = True
        self.fixing_segmentation_mode = FIX_SEGMENTATION_OFF
        self.fix_segmentation_last_position = None
        # self.working_directory = "c:\\"
        self.working_directory = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2021-12-12-p0_utricle_ablation\\"

    def closeEvent(self, event):
        if self.data_lost_warning():
            del self.tissue_info
            event.accept()  # let the window close
        else:
            event.ignore()

    def connect_methods(self):
        self.zo_check_box.stateChanged.connect(self.zo_related_widget_changed)
        self.atoh_check_box.stateChanged.connect(self.atoh_related_widget_changed)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        self.frame_line_edit.textChanged.connect(self.frame_line_edit_changed)
        self.zo_spin_box.valueChanged.connect(self.zo_related_widget_changed)
        self.atoh_spin_box.valueChanged.connect(self.atoh_related_widget_changed)
        self.zo_level_scroll_bar.valueChanged.connect(self.zo_related_widget_changed)
        self.atoh_level_scroll_bar.valueChanged.connect(self.atoh_related_widget_changed)
        self.segment_frame_button.clicked.connect(self.segment_frame)
        self.open_file_pb.clicked.connect(self.open_file)
        self.segment_all_frames_button.clicked.connect(self.segment_all_frames)
        self.show_segmentation_check_box.stateChanged.connect(self.segmentation_related_widget_changed)
        self.show_cell_types_check_box.stateChanged.connect(self.analysis_related_widget_changed)
        self.show_cell_tracking_check_box.stateChanged.connect(self.analysis_related_widget_changed)
        self.cell_tracking_spin_box.valueChanged.connect(self.analysis_related_widget_changed)
        self.show_neighbors_check_box.stateChanged.connect(self.analysis_related_widget_changed)
        self.saveFile.activated.connect(self.save_data)
        self.undo.activated.connect(self.undo_last_action)
        self.save_segmentation_button.clicked.connect(self.save_data)
        self.load_segmentation_button.clicked.connect(self.load_data)
        self.analyze_segmentation_button.clicked.connect(self.analyze_segmentation)
        self.track_cells_button.clicked.connect(self.track_cells)
        self.cancel_segmentation_button.clicked.connect(self.cancel_segmentation)
        self.cancel_analysis_button.clicked.connect(self.cancel_analysis)
        self.cancel_tracking_button.clicked.connect(self.cancel_tracking)
        self.plot_single_cell_data_button.clicked.connect(self.plot_single_cell_data)
        self.image_display.photoClicked.connect(self.image_clicked)
        self.fix_segmentation_button.clicked.connect(self.fix_segmentation)
        self.finish_fixing_segmentation_button.clicked.connect(self.finish_fixing_segmentation)

    def open_file(self):
        global img
        fname = QFileDialog.getOpenFileName(caption='Open File',
                                            directory=self.working_directory, filter="images (*.czi, *.tif)")
        if os.path.isdir(fname[0]):
            return 0
        try:
            self.img, self.img_dimensions, self.img_metadata = read_virtual_image(fname[0])
        except PermissionError or ValueError:
            message_box = QMessageBox
            message_box.about(self,'', 'An error has occurd while oppening file %s' % fname[0])
            return 0
        self.working_directory = os.path.dirname(fname[0])
        self.number_of_frames = self.img_dimensions.T
        self.zo_changed = True
        self.atoh_changed = True
        self.atoh_spin_box.setMaximum(self.img_dimensions.C-1)
        self.zo_spin_box.setMaximum(self.img_dimensions.C-1)
        self.frame_slider.setMaximum(self.number_of_frames)
        self.current_frame = np.zeros((3, self.img_dimensions.X, self.img_dimensions.Y), dtype="uint8")
        self.tissue_info = Tissue(self.number_of_frames, fname[0])
        self.frame_line_edit.setText("%d/%d" % (self.frame_slider.value(), self.number_of_frames))
        self.display_frame()
        self.setState(image=True)

    def display_frame(self):
        frame_number = self.frame_slider.value()
        if self.zo_changed:
            if self.zo_check_box.isChecked():
                zo_channel = self.zo_spin_box.value()
                self.current_frame[1, :, :] = 0.01*self.zo_level_scroll_bar.value()*self.img[frame_number - 1, zo_channel, 0, :, :].compute()
            else:
                self.current_frame[1, :, :] = 0
            self.zo_changed = False
        if self.atoh_changed:
            if self.atoh_check_box.isChecked():
                atoh_channel = self.atoh_spin_box.value()
                self.current_frame[2, :, :] = 0.01*self.atoh_level_scroll_bar.value()*self.img[frame_number - 1, atoh_channel, 0, :, :].compute()
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
                                                 self.cell_tracking_spin_box.value())
            if analysis_img is not None:
                add_analysis = True
        if add_analysis:
            disp_image = np.transpose(np.where(analysis_img == 0, self.current_frame,
                                              np.round(analysis_img*255).astype("uint8")), (1, 2, 0))
        else:
            disp_image = np.transpose(self.current_frame, (1, 2, 0))
        disp_image = cv2.cvtColor(disp_image, cv2.COLOR_BGR2RGB)
        QI = QImage(disp_image, self.img_dimensions.X, self.img_dimensions.Y, QImage.Format_RGB888)
        QI.setColorTable(COLORTABLE)
        self.image_display.setPhoto(QPixmap.fromImage(QI))
        self.display_histogram()

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
        if self.current_segmentation is None:
            self.show_segmentation_check_box.setEnabled(False)
            self.cells_number_changed()

    def cells_number_changed(self):
        cells_num = self.tissue_info.get_cells_number(self.frame_slider.value())
        self.cell_tracking_spin_box.setMaximum(cells_num)
        self.plot_single_cell_data_spin_box.setMaximum(cells_num)

    def update_single_cell_features(self):
        features = self.tissue_info.get_cells_features()
        self.plot_single_cell_data_combo_box.clear()
        self.plot_single_cell_data_combo_box.addItems(features)

    def slider_changed(self):
        text = "%d/%d" % (self.frame_slider.value(), self.number_of_frames)
        if self.frame_line_edit.text() != text:
            self.frame_line_edit.setText(text)
            self.frame_changed()

    def image_clicked(self, click_info):
        pos = click_info.point
        button = click_info.button
        double_click = click_info.doubleClick
        if self.image_display.dragMode() == QGraphicsView.NoDrag:
            if self.fixing_segmentation_mode == FIX_SEGMENTATION_ON:
                frame = self.frame_slider.value()
                if button == Qt.LeftButton:
                    if double_click:
                        self.fix_segmentation_last_position = None
                        self.tissue_info.add_segmentation_line(frame, (pos.x(), pos.y()), final=True,
                                                               hc_marker_image=self.img[frame - 1,
                                                                               self.atoh_spin_box.value(),
                                                                               0, :, :].compute())
                    else:
                        self.tissue_info.add_segmentation_line(frame, (pos.x(), pos.y()),
                                                               self.fix_segmentation_last_position,
                                                               initial=(self.fix_segmentation_last_position is None))
                        self.fix_segmentation_last_position = (pos.x(), pos.y())
                elif button == Qt.MiddleButton:
                    self.tissue_info.remove_segmentation_line(frame, (pos.x(), pos.y()),
                                                              hc_marker_image=self.img[frame - 1,
                                                                                       self.atoh_spin_box.value(),
                                                                                       0, :, :].compute())
                self.segmentation_changed = True
                self.current_segmentation = self.tissue_info.get_segmentation(frame)
                self.display_frame()
            if self.pixel_info.isEnabled():
                text = 'pixel info: x = %d, y = %d' % (pos.x(), pos.y())
                cell = self.tissue_info.get_cell_by_pixel(pos.x(), pos.y(), self.frame_slider.value())
                if cell is not None:
                    if cell.empty:
                        cell_id = 0
                    else:
                        cell_id = cell.label
                    text += '\ncell id = %d' % cell_id
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
        self.display_frame()

    def atoh_related_widget_changed(self):
        self.atoh_changed = True
        self.display_frame()

    def segmentation_related_widget_changed(self):
        self.segmentation_changed = True
        self.display_frame()

    def analysis_related_widget_changed(self):
        self.analysis_changed = True
        self.display_frame()

    def hide_progress_bars(self):
        self.segment_all_frames_progress_bar.hide()
        self.analyze_segmentation_progress_bar.hide()
        self.save_data_progress_bar.hide()
        self.load_data_progress_bar.hide()
        self.track_cells_progress_bar.hide()
        self.cancel_segmentation_button.hide()
        self.cancel_analysis_button.hide()
        self.cancel_tracking_button.hide()
        self.finish_fixing_segmentation_button.hide()
        self.fix_segmentation_label.hide()

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
            self.segmentation_threshold_label.setEnabled(True)
            self.segmentation_kernel_std_label.setEnabled(True)
            self.load_segmentation_button.setEnabled(True)
            self.pixel_info.setEnabled(True)
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
            self.segmentation_threshold_spin_box.setEnabled(False)
            self.segmentation_kernel_std_spin_box.setEnabled(False)
            self.segmentation_threshold_label.setEnabled(False)
            self.segmentation_kernel_std_label.setEnabled(False)
            self.load_segmentation_button.setEnabled(False)
            self.track_cells_button.setEnabled(False)
            self.plot_single_cell_data_spin_box.setEnabled(False)
            self.plot_single_cell_data_combo_box.setEnabled(False)
            self.plot_single_cell_data_button.setEnabled(False)
            self.pixel_info.setEnabled(False)
            self.finish_fixing_segmentation_button.setEnabled(False)
            self.fix_segmentation_label.setEnabled(False)
            self.fix_segmentation_button.setEnabled(False)
        if segmentation:
            self.show_segmentation_check_box.setEnabled(True)
            self.save_segmentation_button.setEnabled(True)
            self.analyze_segmentation_button.setEnabled(True)
            self.fix_segmentation_button.setEnabled(True)
        else:
            self.show_segmentation_check_box.setEnabled(False)
            self.save_segmentation_button.setEnabled(False)
            self.analyze_segmentation_button.setEnabled(False)
            analysis = False
        if analysis:
            self.show_cell_types_check_box.setEnabled(True)
            self.show_cell_tracking_check_box.setEnabled(True)
            self.cell_tracking_spin_box.setEnabled(True)
            self.show_neighbors_check_box.setEnabled(True)
            self.track_cells_button.setEnabled(True)
            self.plot_single_cell_data_spin_box.setEnabled(True)
            self.plot_single_cell_data_combo_box.setEnabled(True)
            self.plot_single_cell_data_button.setEnabled(True)
        else:
            self.show_cell_types_check_box.setEnabled(False)
            self.show_cell_tracking_check_box.setEnabled(False)
            self.cell_tracking_spin_box.setEnabled(False)
            self.show_neighbors_check_box.setEnabled(False)
            self.track_cells_button.setEnabled(False)
            self.plot_single_cell_data_spin_box.setEnabled(False)
            self.plot_single_cell_data_combo_box.setEnabled(False)
            self.plot_single_cell_data_button.setEnabled(False)


    def plot_single_cell_data(self):
        if self.plot_single_cell_data_combo_box.currentIndex() < 0:
            return 0
        cell_id = self.plot_single_cell_data_spin_box.value()
        feature = self.plot_single_cell_data_combo_box.currentText()
        self.tissue_info.plot_single_cell_data(cell_id, feature)

    def frame_segmentation_done(self, msg):
        split_msg = msg.split("/")
        frame = int(split_msg[0])
        percentage_done = int(split_msg[1])
        self.current_segmentation = self.tissue_info.get_segmentation(frame)
        self.segmentation_changed = True
        self.segmentation_saved = False
        self.frame_slider.setValue(frame)
        self.segment_all_frames_progress_bar.setValue(percentage_done)
        self.display_frame()
        if percentage_done == 100:
            self.frame_slider.setEnabled(True)
            self.frame_line_edit.setEnabled(True)
            self.segment_all_frames_button.setEnabled(True)
            self.cancel_segmentation_button.hide()
            self.segment_all_frames_progress_bar.hide()
            self.setState(image=True, segmentation=True)

    def segment_frames(self, frame_numbers):
        self.segment_all_frames_progress_bar.reset()
        self.segment_all_frames_progress_bar.show()
        self.cancel_segmentation_button.show()
        zo_channel = self.zo_spin_box.value()
        threshold = 0.01 * self.segmentation_threshold_spin_box.value()
        std = self.segmentation_kernel_std_spin_box.value()
        self.segmentation_thread = SegmentAllThread(self.img, zo_channel, threshold, std, frame_numbers, self.tissue_info)
        self.segmentation_thread._signal.connect(self.frame_segmentation_done)
        self.segment_all_frames_button.setEnabled(False)
        self.frame_slider.setEnabled(False)
        self.frame_line_edit.setEnabled(False)
        self.segmentation_thread.start()


    def segment_frame(self, frame_number=0):
        if frame_number == 0:
            frame_number = self.frame_slider.value()
        else:
            self.frame_slider.setValue(frame_number)
        self.segment_frames([frame_number])

    def this_might_take_a_while_message(self):
        message_box = QMessageBox
        ret = message_box.question(self, '', "Are you sure? this might take a while...",
                                   message_box.Yes | message_box.No)
        return ret == message_box.Yes

    def segment_all_frames(self):
        if not self.data_lost_warning():
            return 0
        if self.this_might_take_a_while_message():
            self.segment_frames(np.arange(1,self.number_of_frames+1))

    def cancel_segmentation(self):
        self.segmentation_thread.kill()

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
            self.frame_slider.setEnabled(True)
            self.frame_line_edit.setEnabled(True)

    def analyze_frames(self, frame_numbers):
        self.analyze_segmentation_progress_bar.reset()
        self.analyze_segmentation_progress_bar.show()
        self.cancel_analysis_button.show()
        atoh_channel = self.atoh_spin_box.value()
        self.analysis_thread = AnalysisThread(self.img, self.tissue_info, frame_numbers, atoh_channel)
        self.analysis_thread._signal.connect(self.frame_analysis_done)
        self.analyze_segmentation_button.setEnabled(False)
        self.frame_slider.setEnabled(False)
        self.frame_line_edit.setEnabled(False)
        self.analysis_thread.start()

    def analyze_segmentation(self):
        if not self.data_lost_warning():
            return 0
        if self.this_might_take_a_while_message():
            self.analyze_frames(np.arange(1, self.number_of_frames+1))

    def cancel_analysis(self):
        self.analysis_thread.kill()

    def get_analysis_img(self, types, neighbors, track, track_cell_label=0):
        frame_number = self.frame_slider.value()
        if types or neighbors or track:
            img = np.zeros(self.current_frame.shape)
            if types:
                img += self.tissue_info.draw_cell_types(frame_number)
            if neighbors:
                img += self.tissue_info.draw_neighbors_connections(frame_number)
            if track and track_cell_label > 0:
                img += self.tissue_info.draw_cell_tracking(frame_number, track_cell_label)
            return np.clip(img, 0, 1)
        else:
            return None

    def cells_tracking_done(self, msg):
        percentage_done = int(msg)
        self.analysis_saved = False
        self.track_cells_progress_bar.setValue(percentage_done)

        if percentage_done == 100:
            self.cancel_tracking_button.hide()
            self.track_cells_progress_bar.hide()
            self.setState(image=True, segmentation=True, analysis=True)
            self.frame_slider.setEnabled(True)
            self.frame_line_edit.setEnabled(True)

    def track_cells(self):
        self.track_cells_progress_bar.reset()
        self.track_cells_progress_bar.show()
        self.cancel_tracking_button.show()
        self.tracking_thread = TrackingThread(self.tissue_info)
        self.tracking_thread._signal.connect(self.cells_tracking_done)
        self.track_cells_button.setEnabled(False)
        self.frame_slider.setEnabled(False)
        self.frame_line_edit.setEnabled(False)
        self.tracking_thread.start()

    def cancel_tracking(self):
        self.tracking_thread.kill()

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

    def finish_fixing_segmentation(self):
        self.tissue_info.update_labels(self.frame_slider.value())
        self.segmentation_changed = True
        self.current_segmentation = self.tissue_info.get_segmentation(self.frame_slider.value())
        self.display_frame()
        self.fixing_segmentation_mode = FIX_SEGMENTATION_OFF
        self.fix_segmentation_button.setEnabled(True)
        self.fix_segmentation_button.show()
        self.fix_segmentation_label.setEnabled(False)
        self.fix_segmentation_label.hide()
        self.finish_fixing_segmentation_button.setEnabled(False)
        self.finish_fixing_segmentation_button.hide()
        self.frame_slider.setEnabled(True)
        self.frame_line_edit.setEnabled(True)

    def save_data(self):
        name = QFileDialog.getSaveFileName(self, 'Save File', directory=self.working_directory)[0]
        self.save_data_progress_bar.reset()
        self.save_data_progress_bar.show()
        self.frame_slider.setEnabled(False)
        self.frame_line_edit.setEnabled(False)
        self.save_thread = SaveDataThread(self.tissue_info, name)
        self.save_thread._signal.connect(self.frame_saving_done)
        self.save_thread.start()
        return 0

    def frame_saving_done(self, msg):
        percentage_done = msg
        self.save_data_progress_bar.setValue(percentage_done)
        if percentage_done == 100:
            self.segmentation_saved = True
            self.analysis_saved = True
            self.save_data_progress_bar.hide()
            self.frame_slider.setEnabled(True)
            self.frame_line_edit.setEnabled(True)

    def data_lost_warning(self):
        message_box = QMessageBox
        if not self.segmentation_saved or not self.analysis_saved:
            ret = message_box.question(self, '', "Current data will be lost, do you want to save it first?",
                                       message_box.Save | message_box.Discard | message_box.Cancel)
            if ret == message_box.Cancel:
                return False
            elif ret == message_box.Save:
                self.save_data()
        return True

    def load_data(self):
        if not self.data_lost_warning():
            return 0
        name = QFileDialog.getOpenFileName(caption='Open File',
                                            directory=self.working_directory, filter="*.seg")[0]
        self.load_data_progress_bar.reset()
        self.load_data_progress_bar.show()
        self.load_thread = LoadDataThread(self.tissue_info, name)
        self.load_thread._signal.connect(self.frame_loading_done)
        try:
            self.load_thread.start()
        except shutil.ReadError:
            message_box = QMessageBox
            message_box.about(self, '', 'Could not load file %s' % name[0])
            return 0
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
                          analysis=self.tissue_info.is_any_analyzed())
            self.cells_number_changed()
            self.update_single_cell_features()



class SegmentAllThread(QThread):
    _signal = pyqtSignal(str)

    def __init__(self, img, zo_channel, threshold, std, frame_numbers, out):
        super(SegmentAllThread, self).__init__()
        self.img = img
        self.zo_channel = zo_channel
        self.out = out
        self.frame_numbers = frame_numbers
        self.threshold = threshold
        self.std = std
        self.is_killed = False

    def __del__(self):
        self.wait()

    def run(self):
        done_frames = 0
        for frame in self.frame_numbers:
            zo_img = self.img[frame - 1, self.zo_channel, 0, :, :].compute()
            labels = watershed_segmentation(zo_img, self.threshold*np.max(zo_img), self.std)
            self.out.set_labels(frame, labels)
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


class AnalysisThread(QThread):
    _signal = pyqtSignal(str)

    def __init__(self, img, tissue_info, frame_numbers, atoh_channel):
        super(AnalysisThread, self).__init__()
        self.tissue_info = tissue_info
        self.img = img
        self.frame_numbers = frame_numbers
        self.atoh_channel = atoh_channel
        self.is_killed = False

    def __del__(self):
        self.wait()

    def run(self):
        done_frames = 0
        for frame in self.frame_numbers:
            atoh_img = self.img[frame - 1, self.atoh_channel, 0, :, :].compute()
            self.tissue_info.calculate_frame_cellinfo(frame, atoh_img)
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


class TrackingThread(QThread):
    _signal = pyqtSignal(str)

    def __init__(self, tissue_info):
        super(TrackingThread, self).__init__()
        self.tissue_info = tissue_info
        self.is_killed = False

    def __del__(self):
        self.wait()

    def run(self):
        tracking_generator = self.tissue_info.track_cells()
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


class SaveDataThread(QThread):
    _signal = pyqtSignal(int)

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


class LoadDataThread(QThread):
    _signal = pyqtSignal(int)

    def __init__(self, tissue_info, file_path):
        super(LoadDataThread, self).__init__()
        self.tissue_info = tissue_info
        self.path = file_path

    def __del__(self):
        self.wait()

    def run(self):
        for percent_done in self.tissue_info.load(self.path):
            self.emit(percent_done)
        self.emit(100)

    def emit(self, msg):
        self._signal.emit(int(msg))


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = FormImageProcessing()
    w.show()
    sys.exit(app.exec_())