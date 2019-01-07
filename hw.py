# -*- coding: utf-8 -*-

import sys
from hw_ui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore

from model import *
from data import *
from keras.models import load_model

import numpy as np
import cv2


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    # Write your code below
    # UI components are defined in hw1_ui.py, please take a look.
    # You can also open hw1.ui by qt-designer to check ui components.

    def onBindingUI(self):
        self.pushButton.clicked.connect(self.on_pushButton_click)
        self.pushButton.setText("open image")	


    def on_pushButton_click(self):
        fileName1, filetype = QFileDialog.getOpenFileName(self, "選取測試文件", "./", "All Files(*);;Text Files (*.txt)") 
        

        fileName2, filetype = QFileDialog.getOpenFileName(self, "選取標籤文件", "./", "All Files(*);;Text Files (*.txt)") 
        
        pixmap = QPixmap(fileName1)
        pixmap = pixmap.scaled(200, 200, QtCore.Qt.KeepAspectRatio)
        self.label_10.setPixmap(pixmap)
        pixmap = QPixmap(fileName2)
        pixmap = pixmap.scaled(200, 200, QtCore.Qt.KeepAspectRatio)
        self.label_11.setPixmap(pixmap)


        model = load_model('unet_hw.h5')
        testGene = testGenerator(fileName1, 1)
        results = model.predict_generator(testGene,1,verbose=1)
        saveResult("data/Predict",results)

        pixmap = QPixmap('data/Predict/0_predict.png')
        pixmap = pixmap.scaled(200, 200, QtCore.Qt.KeepAspectRatio)
        self.label_12.setPixmap(pixmap)

        img_truth = cv2.imread(fileName2, cv2.IMREAD_GRAYSCALE)
        img_result = cv2.imread('data/Predict/0_predict.png', cv2.IMREAD_GRAYSCALE)

        pixel_count_t = 0
        pixel_count_r = 0
        pixel_count_i = 0
        for i in range(len(img_truth)):
            for j in range(len(img_truth[0])):
                pixel_count_t += 1 if img_truth[i][j] == 255 else 0
                pixel_count_r += 1 if img_result[i][j] == 255 else  0
                pixel_count_i += 1 if img_truth[i][j] == 255 and img_result[i][j] == 255 else  0
        self.label_7.setText(str(pixel_count_r))
        self.label_8.setText(str(pixel_count_t))
        self.label_9.setText(str(pixel_count_i))
        self.label_14.setText(str(2 * pixel_count_i / (pixel_count_r + pixel_count_t)))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
