from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5 import QtGui

from models.experimental import attempt_load
from utils.torch_utils import select_device

import sys
import os
import cv2
import win   # 界面文件
import pic_pre as pp

import time
class AppWindow(QMainWindow, win.Ui_MainWindow):
    def __init__(self, parent=None):
        super(AppWindow, self).__init__(parent)
        self.setupUi(self)

        self.model_path = ''
        self.pic_path = ''
        self.vedio_path = ''
        # self.showimg()
        # self.showimg_pre()
        self.model = '' # 模型
        self.half = '' # 判断是否用显卡
        self.device ='' # 选择是否用显卡

        self.datatime_timer()
        self.pushButton_load_model.clicked.connect(self.load_model_path)
        self.pushButton_load_pic.clicked.connect(self.load_pic_path)
        self.pushButton_pic_pre.clicked.connect(self.pic_pre)
        self.pushButton_load_vedio.clicked.connect(self.load_vedio_path)
        self.pushButton_video_pre.clicked.connect(self.vedio_pre)
        self.pushButtonopencamera.clicked.connect(self.opencvcamera)
        self.camrea = camerathread()

    def opencvcamera(self):
        capture = cv2.VideoCapture(0)
        if capture is None or not capture.isOpened():
            print('Warning: unable to open video source: ', 0)
        else:
            self.camrea.cap = capture
            self.camrea.model = self.model
            self.camrea.device = self.device
            self.camrea.half = self.half
            self.camrea.sinOut.connect(self.show_pre_camera)
            self.camrea.start()

    def show_pre_camera(self, xinghao):
        if xinghao == 'success':
            res = cv2.resize(self.camrea.cv2_img, (1101, 611), interpolation=cv2.INTER_CUBIC)  # 用cv2.resize设置图片大小
            img2 = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)  # opencv读取的bgr格式图片转换成rgb格式
            _image = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3,
                                  QtGui.QImage.Format_RGB888)  # pyqt5转换成自己能放的图片格式
            self.label_success_camera.setPixmap(QtGui.QPixmap.fromImage(_image))
            # self.label_7.setText('count:'+str(self.camrea.count))
            # self.camrea.count = self.camrea.count +1
            self.camrea.start()
        else:
            self.camrea.count = 0
            self.camrea.cap.release()




    def vedio_pre(self):
        if self.model_path == '' or self.vedio_path == '':
            QMessageBox.warning(None, '警告', '未读取到模型文件或者是视频文件，请检查！', QMessageBox.Ok)
        else:
            # 线程
            self.r = YCthread()
            self.r.cap = cv2.VideoCapture(self.vedio_path)
            self.r.model = self.model
            self.r.device = self.device
            self.r.half = self.half
            self.r.sinOut.connect(self.show_pre)
            self.r.start()

    def show_pre(self, xinhao):

        if xinhao == 'success':
            res = cv2.resize(self.r.cv2_img, (1101, 611), interpolation=cv2.INTER_CUBIC)  # 用cv2.resize设置图片大小
            img2 = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)  # opencv读取的bgr格式图片转换成rgb格式
            _image = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3,
                                  QtGui.QImage.Format_RGB888)  # pyqt5转换成自己能放的图片格式
            self.label_success_2.setPixmap(QtGui.QPixmap.fromImage(_image))
            self.label_7.setText('count:'+str(self.r.count))
            self.r.count = self.r.count +1
            self.r.start()
        else:
            self.r.count = 0
            self.r.cap.release()
            print(xinhao)
    def load_vedio_path(self):
        self.vedio_path = QFileDialog.getOpenFileName(None, "选择文件", os.getcwd(), '*.mp4')[0]
        if self.vedio_path != '':
            self.lineEdit_vedio.setText(os.path.basename(self.vedio_path))

    def pic_pre(self):
        if self.model_path == '' or self.pic_path == '':
            QMessageBox.warning(None, '警告', '未读取到模型文件或者是图片文件，请检查！', QMessageBox.Ok)
        else:
            print('正在处理！')
            cv2charimg= pp.pic_pre(cv2.imread(self.pic_path),self.model, self.device, self.half)
            res = cv2.resize(cv2charimg, (531, 611), interpolation=cv2.INTER_CUBIC)  # 用cv2.resize设置图片大小
            img2 = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)  # opencv读取的bgr格式图片转换成rgb格式
            _image = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3,
                                  QtGui.QImage.Format_RGB888)  # pyqt5转换成自己能放的图片格式
            self.label_success.setPixmap(QtGui.QPixmap.fromImage(_image))
    def load_pic_path(self):
        self.pic_path = QFileDialog.getOpenFileName(None, "选择文件", os.getcwd(), '*.jpg *png')[0]
        if self.pic_path != '':
            self.lineEdit_pic.setText(os.path.basename(self.pic_path))
            self.showimg()

    def load_model_path(self):
        self.label_model_satute.setText('正在载入模型请稍等！')
        self.model_path = QFileDialog.getOpenFileName(None, "选择模型文件", os.getcwd(), '*.pt')[0]
        if self.model_path != '':
            self.lineEdit_mode_path.setText(os.path.basename(self.model_path))
            # 设置 显卡
            self.device = select_device()
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

            self.model = attempt_load(self.model_path, map_location=self.device)  # load FP32 model
            if self.half:
                self.label_gpu.setText('GPU:已启用')
                self.model.half()  # to FP16
            self.label_model_satute.setText('模型载入成功！')
        else:
            self.label_model_satute.setText('请选择需要在入的模型！')
            self.lineEdit_mode_path.setText('')
    def showimg(self):
        result = cv2.imread(self.pic_path)
        res = cv2.resize(result, (531, 611), interpolation=cv2.INTER_CUBIC)  # 用cv2.resize设置图片大小
        img2 = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)  # opencv读取的bgr格式图片转换成rgb格式
        _image = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)  # pyqt5转换成自己能放的图片格式

        self.label_yt.setPixmap(QtGui.QPixmap.fromImage(_image))

    def showimg_pre(self):
        result = cv2.imread('./runs/detect/exp3/bus.jpg')
        res = cv2.resize(result, (531, 611), interpolation=cv2.INTER_CUBIC)  # 用cv2.resize设置图片大小
        img2 = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)  # opencv读取的bgr格式图片转换成rgb格式
        _image = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)  # pyqt5转换成自己能放的图片格式

        self.label_success.setPixmap(QtGui.QPixmap.fromImage(_image))

    # 计时器显示时间
    def datatime_timer(self):
        # timer 计时器  在爬虫窗口打开时打开计时器 手动关闭不会关闭计时器 可以被杀死
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.showtime)
        self.timer.start(1000)

    def showtime(self):
        self.datatime = QtCore.QDateTime.currentDateTime().toString('yyyy-MM-dd HH:mm:ss') # 获取当前时间并且设置显示格式
        # 状态栏显示时间
        self.statusbar.showMessage(self.datatime, 0)
# 多线程类
class YCthread(QThread):
    sinOut = pyqtSignal(str)
    def __init__(self, ):
        super(YCthread, self).__init__()
        self.count = 0
        self.cv2_img = ''
        self.cap = ''
        self.model = ''
        self.device = ''

    def run(self):
        cap = self.cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.count)  # 设置要获取的帧号
        a, b = cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
        NUM = cap.get(cv2.CAP_PROP_FPS)
        print(NUM)
        if a:
            img = pp.pic_pre(b, self.model, self.device, self.half)
            self.cv2_img = img
            self.sinOut.emit('success')
        else:
            self.sinOut.emit('final')

# 多线程类
class camerathread(QThread):
    sinOut = pyqtSignal(str)
    def __init__(self, ):
        super(camerathread, self).__init__()
        self.count = 0
        self.cv2_img = ''
        self.cap = ''
        self.model = ''
        self.device = ''

    def run(self):

        a, b = self.cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
        # NUM = self.cap.get(cv2.CAP_PROP_FPS)
        # print(NUM)
        if a:
            img = pp.pic_pre(b, self.model, self.device, self.half)
            self.cv2_img = img
            self.sinOut.emit('success')
        else:
            self.sinOut.emit('final')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = AppWindow()
    win.show()
    sys.exit(app.exec_())