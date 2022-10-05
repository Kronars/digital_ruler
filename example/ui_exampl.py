#!/usr/bin/python

# -------------------------------
# imports
# -------------------------------

import os, sys, time

import cv2

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QGroupBox,
                               QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget, QSlider, QLCDNumber)


class Thread(QThread):
    updateFrame = Signal(QImage)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.trained_file = None
        self.status = True
        self.cap = True

    def set_file(self, fname):
        # The data comes with the 'opencv-python' module
        self.trained_file = os.path.join(cv2.data.haarcascades, fname)

    def run(self):
        self.cap = cv2.VideoCapture(0)
        
        while self.status:
            cascade = cv2.CascadeClassifier(self.trained_file)
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Reading frame in gray scale to process the pattern
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            #blur = cv2.GaussianBlur(gray_frame, (5,5), 0)
            #canny_frame = cv2.Canny(blur, 55, 100)

            detections = cascade.detectMultiScale(gray_frame, scaleFactor=1.1,
                                                  minNeighbors=5, minSize=(30, 30))

            # Drawing green rectangle around the pattern
            for (x, y, w, h) in detections:
                pos_ori = (x, y)
                pos_end = (x + w, y + h)
                color = (178, 123, 47)
                print(pos_ori, pos_end)
                cv2.rectangle(frame, pos_ori, pos_end, color, 2)

            # Reading the image in RGB to display it
            color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Creating and scaling QImage
            h, w, ch = color_frame.shape
            img = QImage(color_frame.data, w, h, ch * w, QImage.Format_RGB888)
            
            scaled_img = img.scaled(640, 480, Qt.KeepAspectRatio)

            # Emit signal
            self.updateFrame.emit(scaled_img)
        sys.exit(-1)
        
class AnotherWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    window_title: str = "Setting - Camera Control"
    slider_color: int = 0
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.window_title)
        self.setMinimumSize(400, 600)
        layout = QVBoxLayout()
        self.label = QLabel("Another Window")
        self.lcd = QLCDNumber(self)
        self.colorSlider = QSlider(Qt.Horizontal, self)
        self.colorSlider.setMinimum(1)
        self.colorSlider.valueChanged[int].connect(self.num)
        layout.addWidget(self.label)
        layout.addWidget(self.lcd)
        layout.addWidget(self.colorSlider)
        self.setLayout(layout)
        #print(self.slider_color)
        
    def num(self):
        #print(self.colorSlider.value())
        slider_color = self.colorSlider.value()
        self.lcd.display(slider_color)
        #print(slider_color)

class Window(QMainWindow):
    window_title: str = "Main - Camera Control"
    
    def __init__(self):
        super().__init__()
        # Title and dimensions
        self.setWindowTitle(self.window_title)
        #self.setGeometry(0, 0, 800, 500)
        self.setMinimumSize(800, 500)
        self.slider_color = AnotherWindow()

        # Main menu bar
        self.menu = self.menuBar()
        self.menu_file = self.menu.addMenu("File")
        exit = QAction("Exit", self, triggered = QApplication.quit)
        self.menu_file.addAction(exit)

        self.menu_about = self.menu.addMenu("&About")
        about = QAction("About Qt", self, shortcut = QKeySequence(QKeySequence.HelpContents),
                        triggered = QApplication.aboutQt)
        self.menu_about.addAction(about)

        # Create a label for the display camera
        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)

        # Thread in charge of updating the image
        self.th = Thread(self)
        self.th.finished.connect(self.close)
        self.th.updateFrame.connect(self.setImage)

        # Model group
        self.group_model = QGroupBox("Trained model")
        self.group_model.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        model_layout = QHBoxLayout()

        self.combobox = QComboBox()
        for xml_file in os.listdir(cv2.data.haarcascades):
            if xml_file.endswith(".xml"):
                self.combobox.addItem(xml_file)

        model_layout.addWidget(QLabel("File:"), 10)
        model_layout.addWidget(self.combobox, 90)
        self.group_model.setLayout(model_layout)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        self.button1 = QPushButton("Start")
        self.button2 = QPushButton("Stop/Close")
        self.button3 = QPushButton("Setting")
        self.lcd = QLCDNumber(self)
        self.button1.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.button2.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.button3.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        buttons_layout.addWidget(self.button2)
        buttons_layout.addWidget(self.button1)
        buttons_layout.addWidget(self.button3)
        buttons_layout.addWidget(self.lcd)

        right_layout = QHBoxLayout()
        right_layout.addWidget(self.group_model, 1)
        right_layout.addLayout(buttons_layout, 1)

        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(right_layout)

        # Central widget
        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Connections
        self.button1.clicked.connect(self.start)
        self.button2.clicked.connect(self.kill_thread)
        self.button2.setEnabled(False)
        self.button3.clicked.connect(self.show_new_window)
        self.lcd.display(10)
        self.combobox.currentTextChanged.connect(self.set_model)
        
        
    def show_new_window(self, checked):
        self.w = AnotherWindow()
        self.w.show()

    @Slot()
    def set_model(self, text):
        self.th.set_file(text)

    @Slot()
    def kill_thread(self):
        print("Finishing...")
        self.button2.setEnabled(False)
        self.button1.setEnabled(True)
        self.th.cap.release()
        cv2.destroyAllWindows()
        self.status = False
        self.th.terminate()
        # Give time for the thread to finish
        time.sleep(1)

    @Slot()
    def start(self):
        print("Starting...")
        self.button2.setEnabled(True)
        self.button1.setEnabled(False)
        self.th.set_file(self.combobox.currentText())
        self.th.start()

    @Slot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())