import sys
import os
import cv2
import copy
import numpy as np

from src.GDA import process_image_given_eyes_LDA
from src.LogReg import process_image_given_eyes_LogReg
from src.SVM import process_image_given_eyes_SVM

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QStatusBar
from PyQt5.QtWidgets import QToolBar
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QPushButton, QDialogButtonBox, QVBoxLayout, QFormLayout, QLineEdit
from PyQt5.QtGui import QIcon, QPixmap, QImage

class ImageHandler:

    def __init__(self):
        self.image = None

    def set_image(self, image):
        self.image = image

    def get_eyed_image(self, right_eye = [], left_eye = []):
        image = copy.deepcopy(self.image)
        if not (right_eye == []):
            image[right_eye[0]-5:right_eye[0]+5, right_eye[1]-5:right_eye[1]+5, :] = np.array([0, 255, 255])
        if not (left_eye == []):
            image[left_eye[0]-5:left_eye[0]+5, left_eye[1]-5:left_eye[1]+5, :] = np.array([0, 255, 255])
        return image


class Window(QMainWindow):
    """Main Window."""
    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        self.setWindowTitle('Cat V Dog Classifier')
        # self.setCentralWidget(QLabel("I'm the Central Widget"))
        self._createMenu()
        self._createToolBar()
        self._createStatusBar()
        self.setGeometry(40, 40, 1000, 500)
        self._centralWidget = QWidget(self)
        self.setCentralWidget(self._centralWidget)
        self.imagePath = ''
        self.label = QLabel()
        self.label.mousePressEvent = self._getPixel
        self.imageShown = False
        self.showing = False
        self.pickingRight = False
        self.pickingLeft = False
        self.rightEye = []
        self.leftEye = []
        self.imageHandler = ImageHandler()
        self.classifier = process_image_given_eyes_LDA
        # self._selectFile()

    def _show(self):
        self.show()
        self._selectFile()

    def _setImagePath(self, path):
        self.imagePath = path

    def _printPath(self):
        print(self.imagePath)

    def _selectFile(self):
        dir_ = QFileDialog.getOpenFileName(self, 'Select a .jpg file:')
        path = dir_[0]
        if not os.path.exists(path):
            self._createStatusBar("File doesn't exist")
        else:
            if path.endswith(".jpg"):
                self._setImagePath(path)
                self._showImage()
                self._createStatusBar(path)
            else:
                self._createStatusBar("Invalid file")

    def _createMenu(self):
        self.menu = self.menuBar().addMenu("&Classifier")
        self.menu.addAction('&LDA', self._LDA)
        self.menu.addAction('&SMV', self._SVM)
        self.menu.addAction('&Logistic', self._LogReg)
        self.menu = self.menuBar().addMenu("&Help")

    def _LDA(self):
        self.classifier = process_image_given_eyes_LDA

    def _LogReg(self):
        self.classifier = process_image_given_eyes_LogReg

    def _SVM(self):
        self.classifier = process_image_given_eyes_SVM

    def _createToolBar(self):
        tools = QToolBar()
        self.addToolBar(tools)
        tools.addAction('Select image', self._selectFile)
        # tools.addAction('Show image', self._showImage)
        tools.addAction('Right eye', self._pickRight)
        tools.addAction('Left eye', self._pickLeft)
        tools.addAction('Classify', self._classify)

    def _createStatusBar(self, s = "Wellcome"):
        status = QStatusBar()
        status.showMessage(s)
        self.setStatusBar(status)

    def _resize(self):
        self.resize(1000, 500)
        self.show()

    def _pickRight(self):
        if self.imageShown:
            self._createStatusBar("Select right eye")
            self.showing = False
            self.pickingRight = True
            self.pickingLeft = False

    def _pickLeft(self):
        if self.imageShown:
            self._createStatusBar("Select left eye")
            self.showing = False
            self.pickingRight = False
            self.pickingLeft = True

    def _showImage(self):
        self.showing = True
        self.pickingRight = False
        self.pickingLeft = False
        if os.path.exists(self.imagePath):
            self.imageShown = True
            image = cv2.imread(self.imagePath)
            if (image.shape[0] < 500 and image.shape[0] > image.shape[1]):
                image = cv2.resize(image, (int(500*image.shape[1]/image.shape[0]), 500))
            elif (image.shape[1] < 500 and image.shape[1] > image.shape[0]):
                image = cv2.resize(image, (500, int(500*image.shape[0]/image.shape[1])))
            if (image.shape[0] > 1000 and image.shape[0] > image.shape[1]):
                image = cv2.resize(image, (int(1000*image.shape[1]/image.shape[0]), 1000))
            elif (image.shape[1] > 1000 and image.shape[1] > image.shape[0]):
                image = cv2.resize(image, (1000, int(1000*image.shape[0]/image.shape[1])))
            # cv2.imshow('demo', image)
            # print(image.shape)
            self.imageHandler.set_image(image)
            image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(image)
            self.label.setPixmap(pixmap)
            self.resize(pixmap.width(), pixmap.height())
            self.setCentralWidget(self.label)
            self.show()
        else:
            self._createStatusBar("Invalid image path")

    def _showLabel(self):
        helloMsg = QLabel('<h1>Hello World!</h1>', parent=self)
        self.setCentralWidget(helloMsg)
        self.resize(500, 500)
        self.show()

    def _getPixel(self, event):
        if self.pickingRight:
            self.rightEye = [event.pos().y(), event.pos().x()]
        elif self.pickingLeft:
            self.leftEye = [event.pos().y(), event.pos().x()]
        image = self.imageHandler.get_eyed_image(self.rightEye, self.leftEye)
        image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(image)
        self.label.setPixmap(pixmap)
        self.resize(pixmap.width(), pixmap.height())
        self.setCentralWidget(self.label)
        self.show()

    def _classify(self):
        self.showing = True
        self.pickingRight = False
        self.pickingLeft = False
        if self.rightEye == []:
            self._createStatusBar("Right eye not selected")
        elif self.leftEye == []:
            self._createStatusBar("Left eye not selected")
        elif self.rightEye[1] > self.leftEye[1]:
            self._createStatusBar("Invalid eyes")
        else:
            image_class, image = self.classifier(self.imageHandler.get_eyed_image(), self.rightEye, self.leftEye)
            # print('self.rightEye:', self.rightEye, 'self.leftEye:', self.leftEye)
            # print(self.imageHandler.get_eyed_image().shape)
            message = 'Image of class CAT' if image_class == 0 else 'Image of class DOG'
            self._createStatusBar(message)
            # cv2.imshow(message, image)
            # cv2.waitKey(1)
            self.show()

def main():
    print('APLIKACIJA')
    app = QApplication(sys.argv)
    win = Window()
    win._show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
