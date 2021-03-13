import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable

import numpy as np
import qimage2ndarray
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.Qt import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import QApplication

from sargen import sar
import tile_functions


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
		self.flysim_button = None
		self.initUI()

	def initUI(self):
		self.setWindowTitle('Bing map selector')
		self.setGeometry(100, 100, 1000, 800)

		mainMenu = self.menuBar()

		self.flysim_button = QAction('Симуляция', self)
		self.flysim_button.triggered.connect(self.begin_simulate_fly)
		mainMenu.addAction(self.flysim_button)

		help_button = QAction('Помощь', self)
		help_button.triggered.connect(lambda: QMessageBox.information(self, 'Краткая справка', '* Кликните левой кнопкой мышки по карте, чтобы добавить точку маршрута.\r\n* Кликните левой кнопкой мышки по точке маршрута, чтобы её удалить.\r\n* Перетащите точку маршрута с зажатой левой кнопкой мышки.'))
		mainMenu.addAction(help_button)

		with open('bing_selector.html', 'r') as f:
			self.webview.setHtml(f.read())
		self.setCentralWidget(self.webview)

	def begin_simulate_fly(self):
		self.flysim_button.enabled = False
		self.webview.page().runJavaScript("window.path", self.continue_simulate_fly)

	def continue_simulate_fly(self, path):
		if len(path) < 2:
			QMessageBox.critical(self, 'Ошибка симуляции', 'Маршрут должен состоять хотя бы из 2 точек.\r\nКликайте лкм по карте для создания точек маршрута.')
		else:
			MapProcessor(path).show()
		self.flysim_button.enabled = True


class QSecondaryWindow(QMainWindow):
	"""
	Window that does not automatically closed after was opened because of garbage collection.
	"""

	def __init__(self, title: str, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.setWindowTitle(title)
		GarbageCollectionManager.freeze(self)

	def closeEvent(self, *args, **kwargs) -> None:
		GarbageCollectionManager.unfreeze(self)
		return super().closeEvent(*args, **kwargs)


class MapProcessor(QSecondaryWindow):

	def __init__(self, path: List[List[float]]):
		super().__init__('Параметры симуляции')
		self.path = [[x[1], x[0]] for x in path]  # TODO: swap long,lat in next script versions
		self.initUI()

	def initUI(self):
		layout = QVBoxLayout()

		simulation_parameters = [
			ParameterDescription('Высота полёта', 1732, type=float),
			ParameterDescription('Ширина диаграммы направленности (°)', 60, tip='Определяет угол обзора камеры-радара.', type=float),
			ParameterDescription('Скорость полёта', 200, type=float),
			ParameterDescription('Частота съёмки', 0.2, type=float),
			ParameterDescription('Ключ API Bing', 'AjJhQyVMzBNnY6-64Wt0GpVT_MckgYdZYCP5tSOS4mAkhjY1Pso5FEiGN9nNf4et'),
			ParameterDescription('Порывы ветра (%)', 20, tip='Определяет нестабильность съёмки.<br>Установите значение 0 для отключения симуляции ветра.', type=float, formatter=lambda x: x / 100)
		]

		label = QLabel('Координаты маршрута:\r\n' + json.dumps(self.path, ensure_ascii=False, indent=4))
		self.control = ParametersControl(*simulation_parameters)
		button = QPushButton('Симулировать аэросъёмку')
		button.clicked.connect(self.simulate)

		splitter = QSplitter(Qt.Horizontal)
		splitter.addWidget(self.control)
		splitter.addWidget(label)
		layout.addWidget(splitter)
		layout.addWidget(button)
		self.widget = QWidget()
		self.widget.setLayout(layout)
		self.setCentralWidget(self.widget)

	def simulate(self):
		# if (directory:=self.get_save_directory()) is None:
		# 	return
		directory = Path.cwd() / f'{self.path[0]}_{self.path[-1]}'
		self.generate_optic_images(directory)
		self.generate_sar_images(directory)
		QMessageBox.information(self, 'Готово', f'Снимки успешно сгенерированы в папке {directory.absolute()}.')

	def get_save_directory(self) -> Optional[Path]:
		directory = QFileDialog.getExistingDirectory(None, 'Выбери пустую папку для сохранения снимков', f'{self.path[0]}_{self.path[-1]}', QFileDialog.ShowDirsOnly)
		if not directory:
			return
		directory = Path(directory)
		if not directory.is_dir() or len(list(directory.glob('*'))) != 0:
			if QMessageBox.warning(self, 'Ошибка сохранения', f'Папка должна быть пустой, в противном случае возможна потеря файлов. Вы всё равно хотите продолжить в {directory.absolute()}?', QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
				return
		return directory

	def generate_optic_images(self, save_directory: Path):
		step_length = int(self.control['Скорость полёта'] / self.control['Частота съёмки'])
		shooting_radius = int(self.control['Высота полёта'] * math.tan(math.radians(self.control['Ширина диаграммы направленности (°)'] / 2)))
		tile_functions.function_call(step_length, shooting_radius, self.control['Ключ API Bing'], self.path)

	def generate_sar_images(self, save_directory: Path):
		sar_directory = save_directory / 'sar'
		sar_directory.mkdir(exist_ok=True)
		transform = True
		deviation = self.control['Порывы ветра (%)']
		if deviation < 0.0001:
			transform = False
		for file in (save_directory / 'rotated').glob('*.png'):
			sar(file, transform=transform, rotate=False, tilt_deviation=deviation, scale_deviation=deviation).save(sar_directory / file.name)


@dataclass
class ParameterDescription:
	name: str
	default: Any
	type: Optional[Callable[[str], Any]] = None
	formatter: Optional[Callable] = None
	tip: str = None


class ParametersControl(QWidget):

	def __init__(self, *parameters: ParameterDescription):
		super().__init__()
		self.layout = QFormLayout()
		self.setLayout(self.layout)
		self.parameters = self.initControls(*parameters)

	def initControls(self, *parameters: ParameterDescription) -> Dict[str, Tuple[ParameterDescription, QLineEdit]]:
		inputs = {}
		for parameter in parameters:
			label = QLabel(parameter.name)
			input = QLineEdit(str(parameter.default))
			if parameter.tip:
				input.setToolTip(parameter.tip)
			self.layout.addRow(label, input)
			inputs[parameter.name] = (parameter, input)
		return inputs

	def __getitem__(self, name: str) -> Any:
		parameter, input = self.parameters[name]
		value = input.text()
		if parameter.type is not None:
			value = parameter.type(value)
		if parameter.formatter is not None:
			value = parameter.formatter(value)
		return value


class GarbageCollectionManager:
	pointers = []

	@staticmethod
	def freeze(pointer):
		GarbageCollectionManager.pointers.append(pointer)

	@staticmethod
	def unfreeze(pointer):
		GarbageCollectionManager.pointers.remove(pointer)


app = QApplication(sys.argv)
main_window = MapSelector()
main_window.show()
sys.exit(app.exec_())
