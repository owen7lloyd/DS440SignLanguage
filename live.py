import cv2
import sys
import os
from PyQt5 import QtWidgets, QtGui, QtCore, uic
from PyQt5.QtWidgets import QFileDialog
from Main import predict_video_mac

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('main.ui', self)

        self.start_button = self.findChild(QtWidgets.QPushButton, 'startButton')
        self.start_button.clicked.connect(self.start_stop_recording)
        self.prediction_label = self.findChild(QtWidgets.QLabel, 'predictionLabel')
        self.recording = False
        self.video_writer = None
        self.file_name = None

        if not os.path.exists("recordings"):
            os.makedirs("recordings")

        self.capture = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.show()

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))
            img = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(img)
            self.label.setPixmap(pixmap)

    def start_stop_recording(self):
        if not self.recording:
            self.file_name, _ = QFileDialog.getSaveFileName(self, "Save Video", "recordings/", "Video Files (*.mp4)")
            if not self.file_name:
                return

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            frame_size = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_writer = cv2.VideoWriter(self.file_name, fourcc, fps, frame_size)

            self.recording = True
            self.start_button.setText("Stop")
            self.timer.timeout.connect(self.write_frame)
        else:
            self.recording = False
            self.start_button.setText("Start")
            self.timer.timeout.disconnect(self.write_frame)
            self.video_writer.release()

            prediction = self.run_model(self.file_name)
            self.prediction_label.setText(f"Predicted sign: {prediction}")

    def write_frame(self):
        ret, frame = self.capture.read()
        if ret:
            self.video_writer.write(frame)

    def run_model(self, video_file_name):
        result = predict_video_mac(video_file_name)
        return result

    def closeEvent(self, event):
        self.capture.release()
        event.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    sys.exit(app.exec_())