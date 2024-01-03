import os

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from utils.dataload import *
from utils.utils import *


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.filenameLabel = QLabel(self)
        self.canvas = FigureCanvas(plt.Figure())

        self.H = 800
        self.W = 640
        self.model = load_model('predict')
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.initUI()
        self.setLayout(self.layout)
        self.setWindowTitle('Hand X-ray Landmark Detection System')
        self.setGeometry(200, 200, 800, 600)

    def initUI(self):
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        open_image_button = QPushButton("Open Image")
        open_image_button.setFixedHeight(35)
        open_image_button.clicked.connect(self.img_load)

        layout.addWidget(self.filenameLabel, alignment=Qt.AlignCenter)
        layout.addWidget(open_image_button)
        self.layout = layout

    def img_load(self):
        self.fileDir, _ = QFileDialog.getOpenFileName(self, "Open Image", r"dataset/valid/0images",
                                                      self.tr("Image Files (*.jpg)"))
        self.filenameLabel.setText(os.path.basename(self.fileDir))
        self.predict()

    def predict(self):
        img = cv2.imread(self.fileDir)
        self.origin_h, self.origin_w = img.shape[:2]

        self.test_data = DataLoader(dataload(path=self.fileDir, H=self.H, W=self.W, phase='predict'))
        for inputs in self.test_data:
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            pred = get_heatmap(outputs, 1e-3, self.origin_h, self.origin_w).numpy()

            inputs = cv2.resize(inputs[0][0].detach().cpu().numpy(), (self.origin_w, self.origin_h))

            self.canvas.figure.clear()
            ax = self.canvas.figure.add_subplot(111)
            ax.imshow(inputs, cmap='gray')

            for i in pred:
                ax.scatter(int(i[1]), int(i[0]), s=20, marker='.', c='b')

            self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()
