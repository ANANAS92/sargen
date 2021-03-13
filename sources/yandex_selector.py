import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import qimage2ndarray
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.Qt import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import QApplication

from sargen import sar


def pixmap2array(map: QPixmap) -> np.ndarray:
	size = map.size()
	width = size.width()
	height = size.height()
	array = np.zeros((height, width, 3), dtype=float)
	map = map.toImage()
	for x in range(width):
		for y in range(height):
			color = map.pixel(x, y)
			array[y, x, :] = QColor(color).getRgbF()[:3]
	return np.round(array * 255).astype(np.int8)


def array2pixmap(array: np.ndarray) -> QPixmap:
	array = array
	img = QImage(array, array.shape[1], array.shape[0], QImage.Format_RGB888)
	return QPixmap(img)


def array2image(array: np.ndarray) -> Image:
	return Image.fromarray(array)


def image2array(image: Image) -> np.ndarray:
	return np.array(image)


class MapSelector(QMainWindow):


	def __init__(self):
		super().__init__()
		self.webview = QWebEngineView()
		self.map_processors = []
		self.initUI()

	def initUI(self):
		self.setWindowTitle('Yandex map selector')
		self.setGeometry(100, 100, 1000, 800)

		mainMenu = self.menuBar()
		fileMenu = mainMenu.addMenu('Test')

		exitButton = QAction('run js', self)
		# exitButton.setShortcut('Ctrl+Q')
		# exitButton.setStatusTip('Exit application')
		exitButton.triggered.connect(self.runjs_test)
		fileMenu.addAction(exitButton)

		screenshot_button = QAction('take screenshot', self)
		screenshot_button.triggered.connect(self.screenshot)
		mainMenu.addAction(screenshot_button)

		self.webview.load(QUrl("https://yandex.ru/maps?l=sat&ll=30.206634%2C60.089727&z=17"))
		self.setCentralWidget(self.webview)

	def runjs_test(self):
		self.webview.page().runJavaScript("""
window.onload = function (){
css = document.createElement('style');
css.type = 'text/css';
document.head.appendChild(css);
css.innerText = '#app { display: none; }';
alert('done');
};
""")

	def screenshot(self):
		screen = QtWidgets.QApplication.primaryScreen()
		screenshot = screen.grabWindow(self.winId())
		map_processor = MapProcessor(screenshot, str(self.webview.url()), self)
		self.map_processors.append(map_processor)
		map_processor.show()


class MapProcessor(QWidget):


	def __init__(self, map: QPixmap, url: str, map_selector: MapSelector):
		super().__init__()
		self.map_selector = map_selector
		self.parse_url(url)
		self.initUI(self.crop_map(map))

	def crop_map(self, map: QPixmap) -> QPixmap:
		self.array = qimage2ndarray.byte_view(map.toImage())[180:-60, 420:-60, 0:-1][:, :, ::-1]
		return QPixmap.fromImage(qimage2ndarray.array2qimage(self.array))

	def parse_url(self, url: str):
		data = re.search(r'll=(?P<longitude>-?\d+\.\d+)%2C(?P<latitude>-?\d+\.\d+)&z=(?P<scale>\d+)', url)
		if data is None:
			raise ValueError(f'Can not parse {url}.')
		scales = dict(zip(map(str, range(2, 20)), [3e6, 7e5, 4e5, 2e5, 9e4, 5e4, 2e4, 1e4, 6e3, 3e3, 1e3, 7e2, 4e2, 2e2, 90, 40, 20, 10]))
		self.longitude = float(data.group('longitude'))
		self.latitude = float(data.group('latitude'))
		self.scale = scales[data.group('scale')]

	def initUI(self, map: QPixmap):
		layout = QVBoxLayout()

		label = QLabel("")
		label.setPixmap(map)
		self.control = GenerationControl()
		button = QPushButton('Generate')
		button.clicked.connect(self.generate)

		splitter = QSplitter(Qt.Horizontal)
		splitter.addWidget(self.control)
		splitter.addWidget(label)
		layout.addWidget(splitter)
		layout.addWidget(button)
		self.setLayout(layout)

	def close(self) -> bool:
		self.map_selector.map_processors.remove(self)
		return super().close()

	def generate(self):
		directory = QFileDialog.getExistingDirectory(None, 'Выбери пустую папку для сохранения снимков', '.', QFileDialog.ShowDirsOnly)
		if not directory:
			return
		directory = Path(directory)
		if not directory.is_dir() or len(list(directory.glob('*'))) != 0:
			if QMessageBox.warning(self, 'Ошибка сохранения', f'Папка должна быть пустой, в противном случае возможна потеря файлов. Вы всё равно хотите продолжить в {directory.absolute()}?', QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
				return
		Image.fromarray(self.array).save(directory / 'optic.jpg')
		for i in range(int(self.control.inputs['Количество радиолокационных снимков'].text())):
			sar(self.array).save(directory / f'radio{i + 1}.jpg')
		(directory / 'description.json').write_text(json.dumps({
			'optic': {
				'center_longitude': self.longitude,
				'center_latitude':  self.latitude,
				'scale':            self.scale
			}
		}), 'utf-8')
		QMessageBox.information(self, 'Готово', f'Снимки успешно сгенерированы в папке {directory.absolute()}.')


class GenerationControl(QWidget):


	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		self.layout = QFormLayout()
		self.setLayout(self.layout)
		self.inputs = self.initControls(('Количество радиолокационных снимков', 10))

	def initControls(self, *labels_inputs: Tuple[str, int]) -> Dict[str, QLineEdit]:
		inputs = {}
		for text, value in labels_inputs:
			label = QLabel(text)
			input = QLineEdit(str(value))
			self.layout.addRow(label, input)
			inputs[text] = input
		return inputs


app = QApplication(sys.argv)
main_window = MapSelector()
main_window.show()
sys.exit(app.exec_())
