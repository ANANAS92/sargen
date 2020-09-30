"""
Unit tests for SAR generator functions.
"""
from typing import Any, Tuple

# noinspection PyProtectedMember
from sargen.sources.sargen import _assert_array
from sargen.sources.sargen import *
from nptyping import NDArray
import numpy as np


def test__assert_array():
	"""
	Validate different types, shapes and arrays' data types.
	"""
	valid_objects = np.array([[1, 2]]), np.array([[1.0, 2.0], [1.2, 2.5]], dtype=np.int32)
	invalid_objects = None, 1, np.array([1, 2]), np.array([[1, 2, 3]]), np.array([[1.0, 2.0], [1.2, 2.5]]), np.array([[1.0, 2.0], [1.2, 2.5]], dtype=np.int64), np.array([[1.0, 2.0], [1.2, 2.5]], dtype=np.int16)
	signature = NDArray[(Any, 2), np.int32]
	for obj in valid_objects:
		# noinspection PyTypeChecker
		assert _assert_array(obj, signature) is None, f'Object {obj} has shape {signature} and must be valid.'
	for obj in invalid_objects:
		exception = None
		try:
			# noinspection PyTypeChecker
			_assert_array(obj, signature)
		except ValueError as error:
			exception = error
		finally:
			assert exception is not None, f'Object {obj} does not have shape {signature} and must be invalid.'


def test_clamp_array():
	"""
	Clamp random numbers of pseudorandom shape as byte.
	"""
	# noinspection PyTypeChecker
	clamped = clamp_array(np.random.default_rng().uniform(-1000, 1000, (1000, 500, 2)), 0, 255)
	assert ((clamped >= 0) & (clamped <= 255)).all()


def test_convert_image():
	"""
	Test output of custom function input.
	"""
	# L = R+1.5*G+2*B+2.5
	weights = np.array([1, 1.5, 2])
	bias = 2.5
	# data as R, G, B, L
	data = np.array([[[0, 0, 0, 2.5], [1, 1, 1, 7]], [[1.5, 0.1, 11e-1, 6.35], [54, 12, 1, 76.5]]])
	# noinspection PyTypeChecker
	assert np.allclose(convert_image(data[:, :, :3], channels_weights=weights, bias=bias), data[:, :, 3])


def test_noise_image():
	"""
	Test distribution over zeroes array.
	"""
	# noinspection PyTypeChecker
	data = noise_image(np.zeros((100, 100), dtype=float), mean=100, deviation=20, generator=np.random.default_rng())
	assert round(data.mean()) == 100 and round(data.std()) == 20


def _find_white_dot(array: NDArray[np.uint8]) -> Tuple[Tuple[int, int], float]:
	"""
	Find white dot in array (0 is black, >0 is white) and return its coordinates and angle in radians (0 - 2Pi).
	"""
	y = array.argmax(axis=0).max()
	x = array.argmax(axis=1).max()
	vector = np.array([x, y]) - np.array(array.shape) / 2  # from center
	angle = math.acos(vector[0] / np.linalg.norm(vector))  # angle between vector and {1; 0}
	if vector[1] < 0:
		angle = 2 * math.pi - angle
	return (x, y), angle


def test_rotate_image():
	"""
	Generate image 100x100 and assert it is rotated uniformly randomly and it's size is not less then 30x30.
	"""
	data = np.zeros((100, 100), dtype=np.uint8)
	data[50, 60] = 255  # set white dot at center + {10; 0} vector
	image = Image.fromarray(data)
	angels = []
	for _ in range(1000):
		# noinspection PyTypeChecker
		rotated_data = np.array(rotate_image(image, generator=np.random.default_rng()))
		size = np.array(rotated_data.shape)
		assert (size >= 30).all(), f'Rotated image must have size (30, 30) but has {rotated_data.shape}.'
		# noinspection PyTypeChecker
		position, angle = _find_white_dot(rotated_data)
		angels.append(math.degrees(angle))
	distribution = np.array(angels)
	# noinspection PyArgumentList
	assert (150 <= distribution.mean() <= 200) and distribution.std() <= 150 and distribution.max() >= 300 and distribution.min() <= 60


def test_scale_image():
	"""
	Generate image 100x100 and assert it is scaled normally randomly.
	"""
	data = np.zeros((100, 100), dtype=np.uint8)
	data[50, 50] = 255  # set white dot at center
	image = Image.fromarray(data)
	sizes = []
	for _ in range(1000):
		# noinspection PyTypeChecker
		scaled_data = np.array(scale_image(image, mean=2, deviation=0.5, generator=np.random.default_rng()))
		sizes.append((scaled_data > 100).sum())
	distribution = np.array(sizes)
	assert (3 <= round(distribution.mean()) <= 6) and (1 <= round(distribution.std()) <= 4)


def test_tilt_image():
	"""
	Tilt will result in scale so test scale.
	"""
	data = np.zeros((100, 100), dtype=np.uint8)
	data[50, 50] = 255  # set white dot at center
	image = Image.fromarray(data)
	sizes = []
	for _ in range(1000):
		# noinspection PyTypeChecker
		tilted_data = np.array(tilt_image(image, deviation=0.3, generator=np.random.default_rng()))
		sizes.append((tilted_data > 100).sum())
	distribution = np.array(sizes)
	assert (1 <= round(distribution.mean()) <= 5) and (1 <= round(distribution.std()) <= 3)


def test_transform_image():
	"""
	Use cases from subtests.
	"""
	data = np.zeros((100, 100), dtype=np.uint8)
	data[50, 60] = 255  # set white dot at center + {10; 0} vector
	image = Image.fromarray(data)
	angels = []
	sizes = []
	for _ in range(1000):
		# noinspection PyTypeChecker
		transformed_data = np.array(transform_image(image, rotate=True, scale=True, tilt=True, scale_mean=2, scale_deviation=0.5, tilt_deviation=0.3, generator=np.random.default_rng()))
		# noinspection PyTypeChecker
		position, angle = _find_white_dot(transformed_data)
		angels.append(math.degrees(angle))
		sizes.append((transformed_data > 100).sum())
	distribution1 = np.array(angels)
	distribution2 = np.array(sizes)
	# noinspection PyArgumentList
	assert (150 <= distribution1.mean() <= 200) and distribution1.std() <= 150 and distribution1.max() >= 300 and distribution1.min() <= 60
	assert (7 <= round(distribution2.mean()) <= 17) and (4 <= round(distribution2.std()) <= 10)


def test_sar():
	"""
	Use cases from subtests.
	"""
	data = np.zeros((100, 100, 3), dtype=np.uint8)
	data[50, 60] = np.array([8, 4, 2], dtype=np.uint8)  # set rgb(80, 80, 80) dot at center + {10; 0} vector
	image = Image.fromarray(data)
	# L=10R+20B+40G+15
	channels_weights = 10, 20, 40
	bias = 15
	angels = []
	sizes = []
	for _ in range(1000):
		# noinspection PyTypeChecker
		sar_data = np.array(sar(image, noise=True, transform=True, channels_weights=channels_weights, bias=bias, noise_mean=10, noise_deviation=2, scale_mean=2, scale_deviation=0.5, tilt_deviation=0.3, generator=np.random.default_rng()))
		# noinspection PyArgumentList
		assert sar_data.max() >= 50  # white color from RGB
		assert (18 <= round(sar_data[:10, :10].mean()) <= 26) and round(sar_data[:10, :10].std()) <= 4  # noise
		# noinspection PyTypeChecker
		position, angle = _find_white_dot(sar_data[:, :, 0])
		angels.append(math.degrees(angle))
		sizes.append((sar_data > 100).sum())
	# noinspection DuplicatedCode
	distribution1 = np.array(angels)
	distribution2 = np.array(sizes)
	# noinspection PyArgumentList
	assert (20 <= distribution1.mean() <= 60) and distribution1.std() <= 10 and distribution1.max() >= 30 and distribution1.min() <= 50
	assert (40 <= round(distribution2.mean()) <= 50) and (20 <= round(distribution2.std()) <= 30)
