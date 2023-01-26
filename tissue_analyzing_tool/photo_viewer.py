from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

class MouseClickInfo(object):
    def __init__(self, point, button, doubleClick=False):
        self.point = point
        self.button = button
        self.doubleClick = doubleClick


class PhotoViewer(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(MouseClickInfo)

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        # self._translation = np.zeros((2,))
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0
            # self._translation = np.zeros((2,))

    def setPhoto(self, pixmap=None):
        has_former_image = self.hasPhoto()
        former_zoom = self._zoom
        # former_translation = self._translation
        self._zoom = 0
        self._translation = np.zeros((2,))
        if pixmap and not pixmap.isNull():
            self._empty = False
            if not has_former_image:
                self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        self.fitInView()
        if has_former_image:
            factor = 1.25**former_zoom if former_zoom > 0 else 0.8**former_zoom
            self.scale(factor, factor)
            # self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
            # self._translate(former_translation[0], former_translation[1])
            # self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
            self._zoom = former_zoom
            # self._translation = former_translation

    def wheelEvent(self, event):
        if self.hasPhoto():
            self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0
            self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            button = event.button()
            if button == QtCore.Qt.RightButton:
                self.toggleDragMode()
            else:
                items = self.items(event.pos())
                point = items[0].mapFromScene(self.mapToScene(event.pos())).toPoint()
                self.photoClicked.emit(MouseClickInfo(point, button))
        super(PhotoViewer, self).mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._photo.isUnderMouse():
            button = event.button()
            if button == QtCore.Qt.LeftButton:
                items = self.items(event.pos())
                point = items[0].mapFromScene(self.mapToScene(event.pos())).toPoint()
                self.photoClicked.emit(MouseClickInfo(point, button, True))
        super(PhotoViewer, self).mousePressEvent(event)

    # def translate(self, dx, dy):
    #     self._translation += np.array([dx, dy])

    # def _translate(self, dx, dy):
    #     p = self.mapToScene(dx, dy)
    #     p0 = self.mapToScene(0, 0)
    #     transform = self.transform()
    #     dx = (p.x() - p0.x())*transform.m11()
    #     dy = (p.y() - p0.y())*transform.m22()
    #     self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
    #     super(PhotoViewer, self).translate(dx, dy)
    #     self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)