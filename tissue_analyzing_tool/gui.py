from PyQt5 import QtCore, uic, QtWidgets, QtGui
import tissue_info
import matplotlib
matplotlib.use('Qt5Agg')
import os.path, shutil
import re
from basic_image_manipulations import *
from tissue_info import Tissue
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import cv2

COLORTABLE=[]
WORKING_DIR = "D:\\Kasirer\\experimental_results\\"
BASEDIR = os.path.dirname(__file__)
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
                self.data.to_pickle(file_path)


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


class FormImageProcessing(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(FormImageProcessing, self).__init__(parent)
        uic.loadUi(os.path.join(BASEDIR, "movie_display.ui"), self)
        self.setWindowTitle("Movie Segmentation")
        self.setState()
        self.saveFile = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        self.undo = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self)
        self.connect_methods()
        self.hide_progress_bars()
        self.img = None
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

    def closeEvent(self, event):
        if self.data_lost_warning(self.close):
            del self.tissue_info
            event.accept()  # let the window close
        else:
            event.ignore()

    def close(self):
        del self.tissue_info
        super(FormImageProcessing, self).close()

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
        self.analyze_frame_button.clicked.connect(self.analyze_frame)
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
        self.save_segmentation_button.clicked.connect(self.save_data)
        self.load_segmentation_button.clicked.connect(self.load_data)
        self.analyze_segmentation_button.clicked.connect(self.analyze_segmentation)
        self.track_cells_button.clicked.connect(self.track_cells)
        self.cancel_segmentation_button.clicked.connect(self.cancel_segmentation)
        self.cancel_analysis_button.clicked.connect(self.cancel_analysis)
        self.cancel_tracking_button.clicked.connect(self.cancel_tracking)
        self.plot_single_cell_data_button.clicked.connect(self.plot_single_cell_data)
        self.plot_single_frame_data_button.clicked.connect(self.plot_single_frame_data)
        self.plot_compare_frames_button.clicked.connect(self.plot_compare_frames_data)
        self.image_display.photoClicked.connect(self.image_clicked)
        self.fix_segmentation_button.clicked.connect(self.fix_segmentation)
        self.finish_fixing_segmentation_button.clicked.connect(self.finish_fixing_segmentation)
        self.fix_cell_types_button.clicked.connect(self.fix_cell_types)
        self.finish_fixing_cell_types_button.clicked.connect(self.finish_fixing_cell_types)
        self.fix_tracking_button.clicked.connect(self.correct_tracking)
        self.mark_event_button.clicked.connect(self.mark_event)
        self.mark_event_combo_box.clear()
        self.mark_event_combo_box.addItems(tissue_info.EVENT_TYPES)
        self.abort_event_marking_button.clicked.connect(self.abort_event_marking)
        self.valid_frame_check_box.stateChanged.connect(self.change_frame_validity)

    def open_file(self):
        global img
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
        self.zo_changed = True
        self.atoh_changed = True
        self.atoh_spin_box.setMaximum(self.img_dimensions.C-1)
        self.zo_spin_box.setMaximum(self.img_dimensions.C-1)
        self.frame_slider.setMaximum(self.number_of_frames)
        self.current_frame = np.zeros((3, self.img_dimensions.X, self.img_dimensions.Y), dtype="uint8")
        self.tissue_info = Tissue(self.number_of_frames, fname)
        self.frame_line_edit.setText("%d/%d" % (self.frame_slider.value(), self.number_of_frames))
        self.setWindowTitle(fname)
        self.display_frame()
        self.setState(image=True)

    def display_frame(self):
        frame_number = self.frame_slider.value()
        if self.zo_changed:
            if self.zo_check_box.isChecked():
                zo_channel = self.zo_spin_box.value()
                if self.img_in_memory:
                    self.current_frame[1, :, :] = 0.1 * self.zo_level_scroll_bar.value() * self.img[frame_number - 1,
                                                                                           zo_channel, 0, :, :]
                else:
                    self.current_frame[1, :, :] = 0.1*self.zo_level_scroll_bar.value()*self.img[frame_number - 1, zo_channel, 0, :, :].compute()
            else:
                self.current_frame[1, :, :] = 0
            self.zo_changed = False
        if self.atoh_changed:
            if self.atoh_check_box.isChecked():
                atoh_channel = self.atoh_spin_box.value()
                if self.img_in_memory:
                    self.current_frame[2, :, :] = 0.01 * self.atoh_level_scroll_bar.value() * self.img[frame_number - 1,
                                                                                              atoh_channel, 0, :, :]
                else:
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
                                                 self.cell_tracking_spin_box.value(),
                                                 self.show_events_check_box.isChecked())
            if analysis_img is not None:
                add_analysis = True
        if add_analysis:
            disp_image = np.transpose(np.where(analysis_img == 0, self.current_frame,
                                              np.round(analysis_img*255).astype("uint8")), (1, 2, 0))
        else:
            disp_image = np.transpose(self.current_frame, (1, 2, 0))
        disp_image = cv2.cvtColor(disp_image, cv2.COLOR_BGR2RGB)
        QI = QtGui.QImage(bytes(disp_image), self.img_dimensions.Y, self.img_dimensions.X, 3*self.img_dimensions.Y, QtGui.QImage.Format_RGB888)
        QI.setColorTable(COLORTABLE)
        self.image_display.setPhoto(QtGui.QPixmap.fromImage(QI))
        self.display_histogram()
        valid_frame = self.tissue_info.is_valid_frame(frame_number)
        self.valid_frame_check_box.setChecked(valid_frame)

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
        if self.current_segmentation is None:
            self.show_segmentation_check_box.setEnabled(False)

    def change_frame_validity(self):
        frame_number = self.frame_slider.value()
        valid = self.valid_frame_check_box.isChecked()
        self.tissue_info.set_validity_of_frame(frame_number, valid)

    def cells_number_changed(self):
        cells_num = self.tissue_info.get_cells_number()
        self.cell_tracking_spin_box.setMaximum(cells_num)
        self.plot_single_cell_data_spin_box.setMaximum(cells_num)

    def update_single_cell_features(self):
        features = self.tissue_info.get_cells_features(self.frame_slider.value())
        self.plot_single_cell_data_combo_box.clear()
        self.plot_single_cell_data_combo_box.addItems(features)

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
        self.plot_single_frame_data_y_combo_box.clear()
        self.plot_single_frame_data_y_combo_box.addItems(features)
        self.plot_single_frame_data_y_combo_box.addItems(self.tissue_info.SPECIAL_FEATURES)
        self.plot_single_frame_data_y_combo_box.addItems(self.tissue_info.SPECIAL_Y_ONLY_FEATURES)
        self.plot_single_frame_data_cell_type_combo_box.clear()
        self.plot_single_frame_data_cell_type_combo_box.addItems(self.tissue_info.CELL_TYPES)
        self.plot_compare_frame_data_cell_type_combo_box.clear()
        self.plot_compare_frame_data_cell_type_combo_box.addItems(self.tissue_info.CELL_TYPES)

    def mark_event(self, pos=None):
        self.mark_event_button.setEnabled(False)
        self.mark_event_button.hide()
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
                                           0, self.event_start_position, (0,0))
                self.mark_event_stage = 0
            else:
                message_box = QtWidgets.QMessageBox
                message_box.about(self, '',
                                  'Go to the last frame of the event\nand click on the relevant cell\n(or cells, in case of division)')
                self.mark_event_stage = 2
        elif self.mark_event_stage == 2:
            self.event_end_position = pos
            self.event_end_frame = self.frame_slider.value()
            if self.mark_event_combo_box.currentText() == "division":
                self.mark_event_stage = 3
            else:
                self.tissue_info.add_event(self.mark_event_combo_box.currentText(), self.event_start_frame,
                                           self.event_end_frame, self.event_start_position, self.event_end_position)
                self.mark_event_stage = 0
        elif self.mark_event_stage == 3:
            self.event_end_frame = self.frame_slider.value()
            self.tissue_info.add_event(self.mark_event_combo_box.currentText(), self.event_start_frame,
                                       self.event_end_frame, self.event_start_position, self.event_end_position,
                                       pos)
            self.mark_event_stage = 0
        if self.mark_event_stage == 0:
            self.abort_event_marking_button.setEnabled(False)
            self.abort_event_marking_button.hide()
            self.mark_event_button.setEnabled(True)
            self.mark_event_button.show()
            self.mark_event_combo_box.setEnabled(True)
            self.analysis_changed = True
            self.display_frame()
        return 0

    def abort_event_marking(self):
        self.mark_event_stage = 0
        self.abort_event_marking_button.setEnabled(False)
        self.abort_event_marking_button.hide()
        self.mark_event_button.setEnabled(True)
        self.mark_event_button.show()
        self.mark_event_combo_box.setEnabled(True)
        return 0

    def slider_changed(self):
        text = "%d/%d" % (self.frame_slider.value(), self.number_of_frames)
        if self.frame_line_edit.text() != text:
            self.frame_line_edit.setText(text)
            self.frame_changed()

    def image_clicked(self, click_info):
        pos = click_info.point
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
                                hc_marker_img = self.img[frame - 1, self.atoh_spin_box.value(), 0, :, :]
                            else:
                                hc_marker_img = self.img[frame - 1, self.atoh_spin_box.value(), 0, :, :].compute()
                            self.tissue_info.add_segmentation_line(frame, (pos.x(), pos.y()), final=True,
                                                                   hc_marker_image=hc_marker_img,
                                                                   hc_threshold=self.hc_threshold_spin_box.value()/100,
                                                                   use_existing_types=self.use_existing_cell_types_check_box.isChecked())
                    else:
                        points_too_far = self.tissue_info.add_segmentation_line(frame, (pos.x(), pos.y()),
                                                               self.fix_segmentation_last_position,
                                                               initial=(self.fix_segmentation_last_position is None))
                        if points_too_far:
                            self.fix_segmentation_last_position = None
                        else:
                            self.fix_segmentation_last_position = (pos.x(), pos.y())
                elif button == QtCore.Qt.MiddleButton:
                    if self.img_in_memory:
                        hc_marker_img = self.img[frame - 1, self.atoh_spin_box.value(), 0, :, :]
                    else:
                        hc_marker_img = self.img[frame - 1, self.atoh_spin_box.value(), 0, :, :].compute()
                    self.tissue_info.remove_segmentation_line(frame, (pos.x(), pos.y()),
                                                              hc_marker_image=hc_marker_img,
                                                              hc_threshold=self.hc_threshold_spin_box.value()/100,
                                                              use_existing_types=self.use_existing_cell_types_check_box.isChecked())
                self.segmentation_changed = True
                self.current_segmentation = self.tissue_info.get_segmentation(frame)
                self.display_frame()
            elif self.fix_cell_types_on:
                frame = self.frame_slider.value()
                if button == QtCore.Qt.LeftButton:
                    self.tissue_info.change_cell_type(frame, (pos.x(), pos.y()))
                elif button == QtCore.Qt.MiddleButton:
                    self.tissue_info.make_invalid_cell(frame, (pos.x(), pos.y()))
                self.analysis_changed = True
                self.display_frame()
            elif self.fix_tracking_on:
                frame = self.frame_slider.value()
                new_label = self.cell_tracking_spin_box.value()
                self.tissue_info.fix_cell_label(frame, (pos.x(), pos.y()), new_label)
                self.track_cells(initial_frame=frame, final_frame=self.number_of_frames)
                self.analysis_changed = True
                self.correct_tracking(off=True)
                self.display_frame()
            elif self.mark_event_stage > 0:
                self.mark_event((pos.x(), pos.y()))
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
        self.cancel_segmentation_button.hide()
        self.cancel_analysis_button.hide()
        self.cancel_tracking_button.hide()
        self.finish_fixing_segmentation_button.hide()
        self.fix_segmentation_label.hide()
        self.finish_fixing_cell_types_button.hide()
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
            self.plot_single_cell_data_spin_box.setEnabled(False)
            self.plot_single_cell_data_combo_box.setEnabled(False)
            self.plot_single_cell_data_button.setEnabled(False)
            self.plot_single_frame_data_button.setEnabled(False)
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
            self.fix_cell_types_label.setEnabled(False)
            self.fix_cell_types_button.setEnabled(False)
            self.mark_event_button.setEnabled(False)
            self.mark_event_combo_box.setEnabled(False)
            self.show_events_check_box.setEnabled(False)
            self.show_events_button.setEnabled(False)
            self.valid_frame_check_box.setEnabled(False)
        if segmentation:
            self.show_segmentation_check_box.setEnabled(True)
            self.save_segmentation_button.setEnabled(True)
            self.analyze_segmentation_button.setEnabled(True)
            self.analyze_frame_button.setEnabled(True)
            self.fix_segmentation_button.setEnabled(True)
            self.hc_threshold_label.setEnabled(True)
            self.hc_threshold_spin_box.setEnabled(True)
        else:
            self.show_segmentation_check_box.setEnabled(False)
            self.save_segmentation_button.setEnabled(False)
            self.analyze_frame_button.setEnabled(False)
            self.analyze_segmentation_button.setEnabled(False)
            self.hc_threshold_label.setEnabled(False)
            self.hc_threshold_spin_box.setEnabled(False)
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
            self.plot_single_frame_data_button.setEnabled(True)
            self.plot_single_frame_data_x_combo_box.setEnabled(True)
            self.plot_single_frame_data_y_combo_box.setEnabled(True)
            self.plot_single_frame_data_cell_type_combo_box.setEnabled(True)
            self.plot_compare_frames_button.setEnabled(True)
            self.compare_frames_line_edit.setEnabled(True)
            self.compare_frames_combo_box.setEnabled(True)
            self.plot_compare_frame_data_cell_type_combo_box.setEnabled(True)
            self.use_existing_cell_types_check_box.setEnabled(True)
            self.fix_cell_types_button.setEnabled(True)
            self.mark_event_button.setEnabled(True)
            self.mark_event_combo_box.setEnabled(True)
            self.show_events_check_box.setEnabled(True)
            self.show_events_button.setEnabled(True)
        else:
            self.show_cell_types_check_box.setEnabled(False)
            self.show_cell_tracking_check_box.setEnabled(False)
            self.cell_tracking_spin_box.setEnabled(False)
            self.show_neighbors_check_box.setEnabled(False)
            self.track_cells_button.setEnabled(False)
            self.plot_single_cell_data_spin_box.setEnabled(False)
            self.plot_single_cell_data_combo_box.setEnabled(False)
            self.plot_single_cell_data_button.setEnabled(False)
            self.plot_single_frame_data_button.setEnabled(False)
            self.plot_single_frame_data_x_combo_box.setEnabled(False)
            self.plot_single_frame_data_y_combo_box.setEnabled(False)
            self.plot_single_frame_data_cell_type_combo_box.setEnabled(False)
            self.plot_compare_frames_button.setEnabled(False)
            self.compare_frames_line_edit.setEnabled(False)
            self.compare_frames_combo_box.setEnabled(False)
            self.plot_compare_frame_data_cell_type_combo_box.setEnabled(False)
            self.use_existing_cell_types_check_box.setEnabled(False)
            self.mark_event_button.setEnabled(False)
            self.mark_event_combo_box.setEnabled(False)
            self.show_events_check_box.setEnabled(False)
            self.show_events_button.setEnabled(False)
            self.abort_event_marking_button.setEnabled(False)
            self.abort_event_marking_button.hide()



    def plot_single_cell_data(self):
        if self.plot_single_cell_data_combo_box.currentIndex() < 0:
            return 0
        cell_id = self.plot_single_cell_data_spin_box.value()
        feature = self.plot_single_cell_data_combo_box.currentText()
        plot_window = PlotDataWindow(self, working_dir=self.working_directory)
        data = self.tissue_info.plot_single_cell_data(cell_id, feature, plot_window.get_ax())
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
        plot_window = PlotDataWindow(self, working_dir=self.working_directory)
        data, error_message = self.tissue_info.plot_single_frame_data(frame, x_feature, y_feature, plot_window.get_ax(),
                                                                      cell_type)
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
        if re.match("(\d+,\s*)*\d(,\s*)?", frames_string):
            frames_list = re.split(",\s*", frames_string)
            frames = [int(f) for f in frames_list]
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


    def segment_frames(self, frame_numbers):
        self.segment_all_frames_progress_bar.reset()
        self.segment_all_frames_progress_bar.show()
        self.cancel_segmentation_button.show()
        zo_channel = self.zo_spin_box.value()
        threshold = 0.01 * self.segmentation_threshold_spin_box.value()
        std = self.segmentation_kernel_std_spin_box.value()
        block_size = self.segmentation_block_size_spin_box.value()
        self.segmentation_thread = SegmentAllThread(self.img, zo_channel, threshold, std, block_size,
                                                    frame_numbers, self.tissue_info, self.img_in_memory)
        self.segmentation_thread._signal.connect(self.frame_segmentation_done)
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
        self.segment_frames([frame_number])

    def this_might_take_a_while_message(self):
        message_box = QtWidgets.QMessageBox
        ret = message_box.question(self, '', "Are you sure? this might take a while...",
                                   message_box.Yes | message_box.No)
        return ret == message_box.Yes

    def segment_all_frames(self):
        if not self.data_lost_warning(self.segment_all_frames):
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
            self.update_single_frame_features()
            self.frame_slider.setEnabled(True)
            self.frame_line_edit.setEnabled(True)
            self.analysis_changed = True
            self.display_frame()

    def analyze_frames(self, frame_numbers):
        self.analyze_segmentation_progress_bar.reset()
        self.analyze_segmentation_progress_bar.show()
        self.cancel_analysis_button.show()
        atoh_channel = self.atoh_spin_box.value()
        hc_threshold = self.hc_threshold_spin_box.value()/100
        self.analysis_thread = AnalysisThread(self.img, self.tissue_info, frame_numbers, atoh_channel, hc_threshold,
                                              self.use_existing_cell_types_check_box.isChecked(), self.img_in_memory)
        self.analysis_thread._signal.connect(self.frame_analysis_done)
        self.analyze_segmentation_button.setEnabled(False)
        self.frame_slider.setEnabled(False)
        self.frame_line_edit.setEnabled(False)
        self.analysis_thread.start()

    def analyze_frame(self, frame_number=0):
        if frame_number == 0:
            frame_number = self.frame_slider.value()
        else:
            self.frame_slider.setValue(frame_number)
        self.analyze_frames([frame_number])

    def analyze_segmentation(self):
        if not self.data_lost_warning(self.analyze_segmentation):
            return 0
        if self.this_might_take_a_while_message():
            self.analyze_frames(np.arange(1, self.number_of_frames+1))

    def cancel_analysis(self):
        self.analysis_thread.kill()

    def get_analysis_img(self, types, neighbors, track, track_cell_label=0, events=False):
        frame_number = self.frame_slider.value()
        if types or neighbors or track or events:
            img = np.zeros(self.current_frame.shape)
            if types:
                img += self.tissue_info.draw_cell_types(frame_number)
            if neighbors:
                img += self.tissue_info.draw_neighbors_connections(frame_number)
            if track and track_cell_label > 0:
                img += self.tissue_info.draw_cell_tracking(frame_number, track_cell_label)
            if events:
                img += self.tissue_info.draw_events(frame_number)
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
        if initial_frame == -1 and final_frame == -1:
            initial_frame = 1
            final_frame = self.number_of_frames
        self.cancel_tracking_button.show()
        if initial_frame >=0 and final_frame > 0:
            img_for_tracking = self.img[initial_frame-1:final_frame, :, :,:,:]
        else:
            img_for_tracking = self.img
        self.tracking_thread = TrackingThread(self.tissue_info, img_for_tracking, self.img_in_memory,
                                              initial_frame=initial_frame, final_frame=final_frame)
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
        self.fix_cell_types_button.setEnabled(False)


    def finish_fixing_segmentation(self):
        frame = self.frame_slider.value()
        if self.fix_segmentation_last_position is not None:
            if self.img_in_memory:
                hc_marker_img = self.img[frame - 1, self.atoh_spin_box.value(), 0, :, :]
            else:
                hc_marker_img = self.img[frame - 1, self.atoh_spin_box.value(), 0, :, :].compute()
            self.tissue_info.add_segmentation_line(frame, self.fix_segmentation_last_position, final=True,
                                                   hc_marker_image=hc_marker_img,
                                                   hc_threshold=self.hc_threshold_spin_box.value()/100,
                                                   use_existing_types=self.use_existing_cell_types_check_box.isChecked())
            self.fix_segmentation_last_position = None
        self.tissue_info.update_labels(self.frame_slider.value())
        # self.track_cells(initial_frame=frame - 1, final_frame=self.number_of_frames)
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
        self.fix_cell_types_button.setEnabled(True)

    def fix_cell_types(self):
        self.fix_cell_types_button.setEnabled(False)
        self.fix_cell_types_button.hide()
        self.fix_cell_types_label.setEnabled(True)
        self.fix_cell_types_label.show()
        self.finish_fixing_cell_types_button.setEnabled(True)
        self.finish_fixing_cell_types_button.show()
        self.frame_slider.setEnabled(False)
        self.frame_line_edit.setEnabled(False)
        self.fix_cell_types_on = True
        self.fix_segmentation_button.setEnabled(False)

    def finish_fixing_cell_types(self):
        self.fix_cell_types_on = False
        self.fix_cell_types_button.setEnabled(True)
        self.fix_cell_types_button.show()
        self.fix_cell_types_label.setEnabled(False)
        self.fix_cell_types_label.hide()
        self.finish_fixing_cell_types_button.setEnabled(False)
        self.finish_fixing_cell_types_button.hide()
        self.frame_slider.setEnabled(True)
        self.frame_line_edit.setEnabled(True)
        self.fix_segmentation_button.setEnabled(True)

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
                self.save_data()
                self.waiting_for_data_save.append(calling_function)
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
            self.load_thread = LoadDataThread(self.tissue_info, name)
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
                          analysis=self.tissue_info.is_any_analyzed())
            self.cells_number_changed()
            self.update_single_cell_features()
            self.update_single_frame_features()



class SegmentAllThread(QtCore.QThread):
    _signal = QtCore.pyqtSignal(str)

    def __init__(self, img, zo_channel, threshold, std, block_size, frame_numbers, out, img_in_memory=False):
        super(SegmentAllThread, self).__init__()
        self.img = img
        self.zo_channel = zo_channel
        self.out = out
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
                zo_img = self.img[frame - 1, self.zo_channel, 0, :, :]
            else:
                zo_img = self.img[frame - 1, self.zo_channel, 0, :, :].compute()
            labels = watershed_segmentation(zo_img, self.threshold, self.std, self.block_size)
            self.out.set_labels(frame, labels, reset_data=True)
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


class AnalysisThread(QtCore.QThread):
    _signal = QtCore.pyqtSignal(str)

    def __init__(self, img, tissue_info, frame_numbers, atoh_channel, hc_threshold, use_existing_types=False,
                 img_in_memory=False):
        super(AnalysisThread, self).__init__()
        self.tissue_info = tissue_info
        self.img = img
        self.frame_numbers = frame_numbers
        self.atoh_channel = atoh_channel
        self.hc_threshold = hc_threshold
        self.is_killed = False
        self.use_existing_types = use_existing_types
        self.img_in_memory = img_in_memory

    def __del__(self):
        self.wait()

    def run(self):
        done_frames = 0
        for frame in self.frame_numbers:
            if self.img_in_memory:
                atoh_img = self.img[frame - 1, self.atoh_channel, 0, :, :]
            else:
                atoh_img = self.img[frame - 1, self.atoh_channel, 0, :, :].compute()
            self.tissue_info.calculate_frame_cellinfo(frame, atoh_img, self.hc_threshold, self.use_existing_types)
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
        tracking_generator = self.tissue_info.track_cells_iterator(initial_frame=self.initial_frame,
                                                                   final_frame=self.final_frame,
                                                                   images=self.images,
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

    app = QtWidgets.QApplication(sys.argv)
    w = FormImageProcessing()
    w.show()
    sys.exit(app.exec_())