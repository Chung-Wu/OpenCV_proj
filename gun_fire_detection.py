from gettext import translation
import sys
import os
import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PIL import Image
from glob import glob
import random


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('gun_fire_detection')
        self.setGeometry(300, 300, 300, 400)

        self.mybutton1 = QPushButton('Load video', self)
        self.mybutton1.setGeometry(50, 50, 200, 50)
        self.mybutton1.clicked.connect(self.select_video)

        self.mybutton2 = QPushButton('Video detect', self)
        self.mybutton2.setGeometry(50, 150, 200, 50)
        self.mybutton2.clicked.connect(self.video_detect)

        self.mybutton3 = QPushButton('WebCam detect', self)
        self.mybutton3.setGeometry(50, 250, 200, 50)
        self.mybutton3.clicked.connect(self.webcam_detect)

        self.mylabel1 = QLabel('', self)
        self.mylabel1.setGeometry(400, 70, 800, 600)

        self.mylabel2 = QLabel('', self)
        self.mylabel2.setGeometry(50, 200, 200, 50)

    def select_video(self):
        self.video_path, filetype = QFileDialog.getOpenFileName(
            self, '開啟檔案', os.getcwd(), 'MP4 Files (*.mp4);;All Files (*)')
        self.mylabel2.setText("")

    def load_yolo(self):
        weights = glob(os.path.join(
            os.getcwd(), "yolov3", "yolov3.weights"))
        cfg = glob(os.path.join(
            os.getcwd(), "yolov3", "yolov3.cfg"))
        obj = glob(os.path.join(
            os.getcwd(), "yolov3", "obj.names"))
        net = cv2.dnn.readNet(weights[0], cfg[0])
        classes = []
        with open(obj[0], "r") as f:
            classes = [line.strip() for line in f.readlines()]

        layers_names = net.getLayerNames()
        output_layers = [layers_names[i[0]-1]
                         for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        return net, classes, colors, output_layers

    def detect_objects(self, img, net, outputLayers):
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(
            320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(outputLayers)
        return blob, outputs

    def get_box_dimensions(self, outputs, height, width):
        boxes = []
        confs = []
        class_ids = []
        for output in outputs:
            for detect in output:
                scores = detect[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]
                if conf > 0.3:
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confs.append(float(conf))
                    class_ids.append(class_id)
        return boxes, confs, class_ids

    def draw_labels(self, boxes, confs, colors, class_ids, classes, img, mode = 'video'):
        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label, (x, y - 5), font, 5, color, 5)
        img = cv2.resize(img, (800, 600))
        if mode == 'webcam':
            cv2.imshow("Image", img)
        '''height, width, channel = img.shape
        bytesPerline = 3 * width
        qImg = QImage(img.data, width, height,
                      bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.mylabel1.setPixmap(QPixmap.fromImage(qImg))'''
        return img

    def webcam_detect(self):
        model, classes, colors, output_layers = self.load_yolo()
        cap = cv2.VideoCapture(1)
        cnt = 0
        while True:
            _, frame = cap.read()
            #if cnt % 10 == 0:
            height, width, channels = frame.shape
            blob, outputs = self.detect_objects(frame, model, output_layers)
            boxes, confs, class_ids = self.get_box_dimensions(outputs, height, width)
            self.draw_labels(boxes, confs, colors, class_ids, classes, frame, mode='webcam')
            '''else:
                frame=cv2.resize(frame, (800,600))
                cv2.imshow("Image", frame)
                cnt +=1'''
            key = cv2.waitKey(1)
            if key == 27:
                break
        cap.release()

    def video_detect(self):
        model, classes, colors, output_layers = self.load_yolo()
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        new_cap = []
        self.mylabel2.setText("Detecting......")
        cnt = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            cnt += 1
            if ret == True:
                # if cnt % 10 == 0:
                height, width, channels = frame.shape
                blob, outputs = self.detect_objects(
                    frame, model, output_layers)
                boxes, confs, class_ids = self.get_box_dimensions(
                    outputs, height, width)
                new_frame = self.draw_labels(
                    boxes, confs, colors, class_ids, classes, frame)
                new_cap.append(new_frame)
                # else:
                # new_cap.append(frame)
                key = cv2.waitKey(1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        self.create_video(new_cap)

    def create_video(self, cap):
        self.mylabel2.setText("Generating......")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (800, 600))
        for frame in cap:
            out.write(frame)
        self.mylabel2.setText("Done")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWidget()
    w.show()
    sys.exit(app.exec_())