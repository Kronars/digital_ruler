from pathlib import Path
import sys, os, time
from math import hypot

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow

from ui_ruler import Ui_MainWindow
from backend.frame_draw import DRAW

class Thread(QThread):
    '''Видеопоток'''
    updateFrame = Signal(QImage)

    def __init__(self, parent, config):
        QThread.__init__(self, parent)
        self.status = False
        self.cap = True
        self.camera_config = config
        
    def run(self):
        self.status = True
        self.cap = cv2.VideoCapture(self.camera_config['id'])
        self.current_frame_rate: int = 0
        fc = 0
        t1 = time.time()

        while self.status:
            ret, frame = self.cap.read()
            fc += 1
            if not ret:
                continue

            color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Creating and scaling QImage
            h, w, ch = color_frame.shape
            img = QImage(color_frame.data, w, h, ch * w, QImage.Format_RGB888)

            self.updateFrame.emit(img)
            
            if fc >= 20:
                self.current_frame_rate = round(fc/(time.time()-t1), 2)
                fc = 0
                t1 = time.time()

class WinVideo(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout()
        self.video = QtWidgets.QLabel()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.video.setSizePolicy(sizePolicy)
        self.video.setAlignment(QtCore.Qt.AlignCenter)
        self.video.setFrameShape(QtWidgets.QFrame.Panel)
        layout.addWidget(self.video)
        self.setLayout(layout)
    
    def showImage(self, image: QPixmap):
        self.video.setPixmap(QPixmap.fromImage(image))


class MainWin(QMainWindow):
    def __init__(self) -> None:
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.camera_config = { # init config
            'id': 0,
            'width': 680,
            'height': 480,
            'frame_rate': 60,
            'fourcc': cv2.VideoWriter_fourcc(*"MJPG")
        }
        # Настройка камеры
        self.th = Thread(self, self.camera_config)
        self.ui.video_start.clicked['bool'].connect(self.switch_video_thread)
        self.ui.camera_source_input.valueChanged.connect(self.update_cam_config)
        self.ui.fourcc_input.currentTextChanged.connect(self.update_cam_config)
        self.ui.w_input.valueChanged.connect(self.update_cam_config)
        self.ui.h_input.valueChanged.connect(self.update_cam_config)

        self.ui.unit_name_input.textChanged.connect(self.update_calibration)
        self.ui.pixel_base_input.valueChanged.connect(self.update_calibration)
        self.ui.cal_range_input.valueChanged.connect(self.update_calibration)
        self.ui.cal_load_path_push.clicked.connect(self.cal_load)
        # Создание доп окон
        self.ui.add_window.clicked.connect(self.create_window)
        self.extra_windows = []

        # Переменные для отрисовки
        self.lock = False    # Первый ли это клик мыши
        self.ui.video.mouseMoveEvent  = self.mouse_event
        self.ui.video.mousePressEvent = self.click_event
        self.mouse_mark = None   # Координаты клика last click (from center)
        self.mouse_raw = (0, 0)  # pixels from top left
        self.mouse_now = (0, 0)  # pixels from center
        self.width = self.camera_config['width']
        self.height = self.camera_config['height']
        self.area = self.width*self.height
        self.cx = int(self.width/2)    # Центр по оси x
        self.cy = int(self.height/2)
        self.dm = hypot(self.cx, self.cy)

        # Переменные для калибровки
        self.unit_suffix: str = 'mm'        
        self.pixel_base:  int = 5
        self.cal_range:   int = 72
        self.cal = dict([(x, self.cal_range/self.dm) for x in range(0, int(self.dm) + 1, self.pixel_base)])
        self.cal_base: int = 5
        self.cal_last: int = None

        # Доступ к файлу калибровки
        self.cur_path = Path(__file__).parent
        self.load_path = Path(self.cur_path, 'main_cal.csv')
        self.save_path = self.cur_path
        self.cal_load()

    def update_cam_config(self):
        self.camera_config = {
            'id': self.ui.camera_source_input.value(),
            'width': self.ui.w_input.value(),
            'height': self.ui.h_input.value(),
            'frame_rate': 30,
            'fourcc': cv2.VideoWriter_fourcc(*self.ui.fourcc_input.currentText())
        }

    # Обновление калибровки
    def cal_load(self):
        load_input = self.ui.cal_load_path_input.text()
        calfile = self.load_path if load_input.__len__() == 0 else load_input
        if os.path.isfile(calfile):
            try:
                with open(calfile) as f:
                    for line in f:
                        line = line.strip()
                        if line and line[0] in ('d',):
                            axis, pixels, scale = [_.strip() for _ in line.split(',', 2)]
                            if axis == 'd':
                                self.cal[int(pixels)] = float(scale)
                self.ui.statusbar.showMessage(f'Калибровочный файл {calfile} успешно загружен')
            except Exception as e:
                self.ui.statusbar.showMessage(f'Ошибка загрузки файла калибровки. Файл - {calfile}, ошибка - {e.args[-1]}')

    def cal_update(self, x, y, unit_distance):
        pixel_base = self.pixel_base
        cal_base   = self.cal_base
        cal_range  = self.cal_range

        pixel_distance = hypot(x, y)
        scale = abs(unit_distance/pixel_distance)
        target = self.baseround(abs(pixel_distance), pixel_base)

        # low-high values in distance
        low  = target*scale - (cal_base/2)
        high = target*scale + (cal_base/2)

        # get low start point in pixels
        start = target
        if unit_distance <= cal_base:
            start = 0
        else:
            while start*scale > low:
                start -= pixel_base

        # get high stop point in pixels
        stop = target
        if unit_distance >= self.baseround(cal_range, pixel_base):
            high = max(self.cal.keys())
        else:
            while stop*scale < high:
                stop += pixel_base

        # set scale
        for x in range(start, stop+1, pixel_base):
            self.cal[x] = scale

    def update_calibration(self):
        '''Обновление переменных из интерфейса'''
        self.unit_suffix = self.ui.unit_name_input.text()
        self.pixel_base  = self.ui.pixel_base_input.value()
        self.cal_range   = self.ui.cal_range_input.value()
        self.cal_last    = self.pixel_base
        self.ui.d_status.setText(f'{self.cal_last} / {self.cal_range}')

    # Запуск остановка видеопотока
    def switch_video_thread(self, state):
        '''Контроль включения и выключения видепотока, его инициализация и отключение'''
        if self.sender().objectName() == 'video_start':
            if state:
                self.sender().setText('Stop')
            else:
                self.sender().setText('Start')
        
        if state:
            self.update_cam_config()
            self.th = Thread(self, self.camera_config)
            self.th.updateFrame.connect(self.setImage)
            self.th.start()
            self.draw = DRAW()
        else:
            self.kill_video_thread()
        
    def click_event(self, e):
        '''Вызывается по клику в пределах видео Задача контролить три состояния клика
        1. ничего не нажато 2. первое нажатие 3. второе нажатие -> 1 пункт'''

        if self.th.status:
            if e.buttons() == QtCore.Qt.LeftButton:
                x, y = e.position().x(), e.position().y()
                ox = x - self.cx
                oy = (y-self.cy)*-1
                if not self.lock:   # Если первый клик
                    if self.mouse_mark:
                        self.lock = True
                    else:
                        self.ox = ox
                        self.oy = oy
                        self.mouse_mark = (ox, oy)
                else:               # Если второй клик
                    self.lock = False
                    self.mouse_now  = (ox, oy)
                    self.mouse_mark = (ox, oy)

            elif e.buttons() == QtCore.Qt.RightButton:
                self.mouse_mark = None

    def mouse_event(self, e):
        '''Собирает данные о местоположении курсора при движении в пределах видеопотока'''
        if self.th.status and not self.lock:
            if e.buttons() == QtCore.Qt.NoButton:
                x, y = e.position().x(), e.position().y()
                self.ox = x - self.cx
                self.oy = (y-self.cy)*-1

                self.mouse_raw = (x, y)
                self.mouse_now = (self.ox, self.oy)

    # Вспомогательные функции
    def baseround(self, x, base=1):
        return int(base * round(float(x)/base))

    def distance(self, x1, y1, x2, y2):
        return hypot(x1-x2, y1-y2)

    def conv(self, x, y):
        d = self.distance(0, 0, x, y)
        scale = self.cal[self.baseround(d, self.pixel_base)]
        return x*scale, y*scale
 
    def create_window(self):
        win = WinVideo()
        self.extra_windows.append(win)
        win.show()

    # Обработка кадра
    def frame_processing(self, image: np.array):
        h, w, *_ = image.shape
        draw = self.draw
        draw.width  = w
        draw.height = h
        text = []
        mx, my = self.mouse_raw   # сокращения что бы не городить self.
        cy, cx = h // 2, w // 2
        self.cy, self.cx = cy, cx

        if self.ui.normilize_box.isChecked():
            cv2.normalize(image, image, self.ui.thresh_min.value(), self.ui.thresh_max.value(), cv2.NORM_MINMAX)

        if self.ui.rotate_box.isChecked():
            image = cv2.rotate(image, cv2.ROTATE_180)

        mode = self.ui.settings.currentWidget().objectName()
        if mode == 'dimension':
            draw.crosshairs(image, 5, weight=2, color='green')
            # mouse cursor lines
            draw.vline(image, self.mouse_raw[0], weight=1, color='green')
            draw.hline(image, self.mouse_raw[1], weight=1, color='green')
            
            if self.mouse_mark:
                # locations
                x1, y1 = self.mouse_mark
                x2, y2 = self.mouse_now
                # convert to distance
                x1c, y1c = self.conv(x1, y1)
                x2c, y2c = self.conv(x2, y2)
                xlen = abs(x1c-x2c)
                ylen = abs(y1c-y2c)
                llen = hypot(xlen, ylen)
                alen = 0
                if max(xlen, ylen) > 0 and min(xlen, ylen)/max(xlen, ylen) >= 0.95:
                    alen = (xlen+ylen)/2
                carea = xlen*ylen

                self.ui.x_len.setText(f'X LEN: {xlen:.2f}{self.unit_suffix}')
                self.ui.y_len.setText(f'Y LEN: {ylen:.2f}{self.unit_suffix}')
                self.ui.l_len.setText(f'L LEN: {llen:.2f}{self.unit_suffix}')
                self.ui.curr_click.setText(f'{self.mouse_now[0]} {self.mouse_now[1]}')
                self.ui.last_click.setText(f'{self.mouse_mark[0]} {self.mouse_mark[1]}')

                # convert to plot locations
                x1 += cx
                x2 += cx
                y1 *= -1
                y2 *= -1
                y1 += cy
                y2 += cy
                x3 = x1+((x2-x1)/2)
                y3 = max(y1, y2)

                # plot
                draw.rect(image, x1, y1, x2, y2, weight=1, color='red')
                draw.line(image, x1, y1, x2, y2, weight=1, color='green')

                # add dimensions
                draw.add_text(image, f'{xlen:.2f}', x1-((x1-x2)/2),
                            min(y1, y2)-8, center=True, color='red')
                draw.add_text(
                    image, f'Area: {carea:.2f}', x3, y3+8, center=True, top=True, color='red')
                if alen:
                    draw.add_text(
                        image, f'Avg: {alen:.2f}', x3, y3+34, center=True, top=True, color='green')
                if x2 <= x1:
                    draw.add_text(image, f'{ylen:.2f}', x1+4,
                                (y1+y2)/2, middle=True, color='red')
                    draw.add_text(image, f'{llen:.2f}',
                                x2-4, y2-4, right=True, color='green')
                else:
                    draw.add_text(
                        image, f'{ylen:.2f}', x1-4, (y1+y2)/2, middle=True, right=True, color='red')
                    draw.add_text(image, f'{llen:.2f}', x2+8, y2-4, color='green')

        elif mode == 'automate':
            self.mouse_mark = None
            auto_percent = self.ui.min_percent_input.value()
            auto_threshold = self.ui.threshld_input.value()
            auto_blur = self.ui.gauss_blur_input.value() | 1

            image_2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_2 = cv2.GaussianBlur(image_2, (auto_blur, auto_blur), 0)

            image_2 = cv2.threshold(image_2, auto_threshold,
                                255, cv2.THRESH_BINARY)[1]

            image_2 = ~image_2    # Инвертация

            # find contours on thresholded image
            contours, nada = cv2.findContours(
                image_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            draw.crosshairs(image, 5, weight = 2, color = 'green')

            # loop over the contours
            for c in contours:
                # contour data (from top left)
                x1, y1, w_r, h_r = cv2.boundingRect(c)
                x2, y2 = x1+w_r, y1+h_r
                x3, y3 = x1+(w_r/2), y1+(h_r/2)

                # percent area
                percent = 100*w_r*h_r/(w * h)

                # if the contour is too small, ignore it
                if percent < auto_percent:
                    continue

                # if the contour is too large, ignore it
                elif percent > 60:
                    continue

                # convert to center, then distance
                x1c, y1c = self.conv(x1-(cx), y1-(cy))
                x2c, y2c = self.conv(x2-(cx), y2-(cy))
                xlen = abs(x1c-x2c)
                ylen = abs(y1c-y2c)
                alen = 0
                if max(xlen, ylen) > 0 and min(xlen, ylen)/max(xlen, ylen) >= 0.95:
                    alen = (xlen+ylen)/2
                carea = xlen*ylen

                # plot
                draw.rect(image, x1, y1, x2, y2, weight=2, color='red')

                # add dimensions
                draw.add_text(image, f'{xlen:.2f}', x1-((x1-x2)/2),
                            min(y1, y2)-8, center=True, color='red')
                draw.add_text(
                    image, f'Area: {carea:.2f}', x3, y2+8, center=True, top=True, color='red')
                if alen:
                    draw.add_text(
                        image, f'Avg: {alen:.2f}', x3, y2+34, center=True, top=True, color='green')
                if x1 < w-x2:
                    draw.add_text(image, f'{ylen:.2f}', x2+4,
                                (y1+y2)/2, middle=True, color='red')
                else:
                    draw.add_text(
                        image, f'{ylen:.2f}', x1-4, (y1+y2)/2, middle=True, right=True, color='red')

        elif mode == 'calibration':
            '''в файле main режим назван config'''
            self.draw.crosshairs(image, 5, weight = 2, color = 'red', invert = True)

            draw.line(image, cx, cy, cx+cx, cy+cy, weight = 1, color = 'red')
            draw.line(image, cx, cy, cx+cy, cy-cx, weight = 1, color = 'red')
            draw.line(image, cx, cy,-cx+cx,-cy+cy, weight = 1, color = 'red')
            draw.line(image, cx, cy, cx-cy, cy+cx, weight = 1, color = 'red')
            # mouse cursor lines (parallel to aligned crosshairs)
            dm = self.dm
            draw.line(image, mx, my, mx+dm, my +
                    (dm*(cy/cx)), weight=1, color  = 'green')
            draw.line(image, mx, my, mx-dm, my -
                    (dm*(cy/cx)), weight=1, color  = 'green')
            draw.line(image, mx, my, mx+dm, my +
                    (dm*(-cx/cy)), weight=1, color = 'green')
            draw.line(image, mx, my, mx-dm, my -
                    (dm*(-cx/cy)), weight=1, color = 'green')
            
            if not self.cal_last:
                self.cal_last = self.cal_base
                self.ui.statusbar.showMessage('Начата калибровка')

            elif self.cal_last <= self.cal_range:
                if self.mouse_mark:
                    self.cal_update(*self.mouse_mark, self.cal_last)
                    self.cal_last += self.cal_base
                    
            else:
                # Сохранение в файл
                try:
                    path = Path(self.ui.cal_save_path_input.text(), 'cal_file.csv')
                    with open(path, 'w') as f:
                        data = list(self.cal.items())
                        data.sort()
                        for key, value in data:
                            f.write(f'd,{key},{value}\n')
                    self.ui.statusbar.showMessage(f'Калибровка успешно записана в директорию {path}')
                except Exception as e:
                    self.ui.statusbar.showMessage(f'Ошибка записи калибровки в файл: {e.args[-1]}')
                finally:
                    self.cal_last = self.cal_base
                    
            self.mouse_mark = None
            self.ui.d_status.setText(f'{self.cal_last} / {self.cal_range}')

        draw.add_text_top_left(image, text)
        return image

    @Slot(QImage)
    def setImage(self, image: QImage):
        '''Приём изображения от камеры -> вызов обработки -> передача в интерфейс'''
        # Предобработка
        image = image.scaled(self.camera_config['width'], self.camera_config['height'], # Если масштабировать изображение после обработки, отображение мыши съедет
        Qt.KeepAspectRatio if self.ui.keep_ratio.isChecked() else Qt.IgnoreAspectRatio)
        # Конвертация из QImage в numpy ndarray
        image  = image.convertToFormat(QtGui.QImage.Format.Format_RGB32)
        h      = image.height()   # приходит в разрешении источника
        w      = image.width()
        ptr    = image.constBits()
        frame  = np.array(ptr).reshape(h, w, 4)
        # Обработка
        try:
            frame = self.frame_processing(frame)
        except Exception as e:
            self.ui.statusbar.showMessage(f'Ошибка отрисовки, разрешение калибровочного файла должно совпадать с текущим разрешением')
        # Конвертация обратно в QImage и отображение
        _, _, ch = frame.shape
        image = QImage(frame.data, w, h, ch * w, QImage.Format_RGB32)
        self.ui.video.setPixmap(QPixmap.fromImage(image))
        # Отображение фреймрейта
        self.ui.fps.setText(self.th.current_frame_rate.__str__())
        # Отображение на дополнительных окнах
        for window in self.extra_windows:
            window.showImage(image)
            

    def kill_video_thread(self):
        print("Shutdown video...")
        self.th.status = False
        self.th.cap.release()
        self.th.terminate()
        # Give time for the thread to finish
        self.ui.video.clear()
        time.sleep(1)

    def closeEvent(self, event):
        print("Exiting...")
        self.kill_video_thread()
        for win in self.extra_windows:
            win.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWin()
    window.show()
    sys.exit(app.exec())