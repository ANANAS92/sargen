"""
Optic to SAR image converter.

Generates radio location 24 bit grayscale (LLL) images from 24 bit RGB optic images with random transformations (noise, rotation, scale,tilt).
"""
import math
from numbers import Number
from pathlib import Path
from typing import Union

import numpy as np  # TODO: update to 1.20, see https://github.com/numpy/numpy/milestone/83
from PIL import Image


def clamp_array(array: np.ndarray, minimum: Number = 0, maximum: Number = 255) -> np.ndarray:
	"""
	Clamp values less then 0 to 0 and greater then 255 to 255.

	:param array: array of numbers.
	:param minimum: minimum value to clamp elements below it.
	:param maximum: maximum value to clamp elements above it.
	:return: arrays of same shape with clamped values.
	"""
	# noinspection PyTypeChecker
	return np.where(array < minimum, minimum, np.where(array > maximum, maximum, array))


def convert_image(image: np.ndarray, *, channels_weights: np.ndarray = np.array([0.299, 0.557, 0.144]), bias: float = -33) -> np.ndarray:
	"""
	Convert R, G, B color channels to luma single channel.

	Note that luma is not clamped.

	:param image: array of pixels of shape (height, width, 3) of floats.
	:param channels_weights: R, G, B coefficients array of shape 3 of floats.
	:param bias: luma bias to apply after conversion.
	:return: array of shape (height, width) of floats.
	"""
	return (image * channels_weights).sum(axis=2) + bias


def noise_image(image: np.ndarray, *, mean: float = 0, deviation: float = 12.5, generator: np.random.Generator = np.random.default_rng()) -> np.ndarray:
	"""
	Add gaussian noise to each pixel of single channel image.

	Note that luma is not clamped.

	:param image: array of pixels of shape (height, width) of floats.
	:param mean: mu parameter of normal distribution.
	:param deviation: sigma parameter of normal distribution also known as standard deviation.
	:param generator: random number generator.
	:return: array of shape (height, width) of floats.
	"""
	return image + generator.normal(mean, deviation, image.shape)


def rotate_image(image: Image.Image, *, generator: np.random.Generator = np.random.default_rng()) -> Image.Image:
	"""
	Rotate image by uniformly random angle.

	Note that image is cropped and converted to RGB (same channels' values in grayscale case).

	:param image: image to rotate and crop.
	:param generator: random number generator.
	"""
	# Rotation over 360 degrees crops image by circle with radius of min(width, height). So we need to crop insquare of that circle.
	angle = generator.uniform(0, 360)
	center = np.array(image.size) / 2
	insquare_side = min(*image.size) / math.sqrt(2)
	insquare = np.concatenate([center - insquare_side / 2, center + insquare_side / 2])
	image = image.rotate(angle, resample=Image.BICUBIC, expand=False, center=tuple(center)).crop(tuple(insquare))
	image.load()  # cropping is lazy so we need to load results
	return image


def scale_image(image: Image.Image, *, deviation: float = 0.2, generator: np.random.Generator = np.random.default_rng()) -> Image.Image:
	"""
	Scale image by gaussian random factor (where mean is 100 %).

	Actually crop image by random factor from (0; 0) pivot point (crop width and height).

	:param image: image to scale.
	:param deviation: sigma parameter of normal distribution also known as standard deviation.
	:param generator: random number generator.
	"""
	factor = 2
	for _ in range(100):
		factor = generator.normal(1, deviation)
		if factor <= 1:
			break
	new_size = (np.array(image.size) * factor).round(0).astype(int)
	image = image.crop((0, 0, *new_size))
	image.load()
	return image


def _find_perspective_coefficients(origin_points: np.ndarray, transformed_points: np.ndarray) -> np.ndarray:
	"""
	Find perspective transformation coefficients a, b, c, d, e, f, g, h.

	See https://stackoverflow.com/a/14178717 along with other answers and comments.

	:param origin_points: 4 origin points that are mapped to transformed (as array of shape (4, 2) of floats).
	:param transformed_points: 4 points to where origin ones are mapped (as array of shape (4, 2) of floats).
	:return: array of shape 8 of floats.
	"""
	matrix_buf = []
	for src_point, dist_point in zip(origin_points, transformed_points):
		matrix_buf.append([src_point[0], src_point[1], 1, 0, 0, 0, -dist_point[0] * src_point[0], -dist_point[0] * src_point[1]])
		matrix_buf.append([0, 0, 0, src_point[0], src_point[1], 1, -dist_point[1] * src_point[0], -dist_point[1] * src_point[1]])

	a_matrix = np.array(matrix_buf, dtype=np.float)
	b_matrix = np.array(transformed_points).reshape(8)

	res = np.linalg.solve(a_matrix, b_matrix)
	return res.reshape(8)


def _shift_corner(corner_points: np.ndarray, index: int, image: Image, deviation: float, generator: np.random.Generator) -> None:
	"""
	Shift bounding box corner outside by random factor.

	:param corner_points: image corner points.
	:param index: index of corner starting from top left (0, 1, 2 or 3).
	:param image: source image.
	:param deviation: maximum perspective factor.
	:param generator: random number generator.
	"""
	if not 0 <= index < 4:
		raise ValueError('Index must be 0, 1, 2 or 3.')
	shift = -generator.uniform(0, deviation * image.size[0], 2).astype(corner_points.dtype)  # inverse for expanding transformation instead of reducing one
	if index == 0 or index == 3:
		shift[0] = -shift[0]
	if index == 0 or index == 1:
		shift[1] = -shift[1]
	corner_points[index] += shift


def tilt_image(image: Image.Image, *, deviation: float = 0.2, generator: np.random.Generator = np.random.default_rng()) -> Image.Image:
	"""
	Tilt image to any direction by gaussian random factor. This is perspective transformation.

	Note that image is cropped to fit origin box.

	:param image: image to tilt and crop.
	:param deviation: maximum perspective factor.
	:param generator: random number generator.
	"""
	# See https://stackoverflow.com/a/14178717 along with other answers and comments.
	# We have 4 points = 8 coordinates that can be shifted (each in one direction only).
	# Tilt can be performed in one or two adjacent points that are shifted randomly. Three points is the same as one points while four points means scale.
	# So we just take 50/50 one or two adjacent corner points and shift them out of bounding box to avoid empty space after perspective.
	origin_points = np.array([[0, 0], [image.width, 0], [image.width, image.height], [0, image.height]])
	transformed_points = origin_points.copy()
	shifted_corner = generator.integers(0, 4)
	_shift_corner(transformed_points, shifted_corner, image, deviation, generator)
	if generator.random() < 0.5:
		shifted_corner = (shifted_corner + 1 if generator.random() < 0.5 else shifted_corner - 1) % 4
		_shift_corner(transformed_points, shifted_corner, image, deviation, generator)
	transformation_parameters = _find_perspective_coefficients(origin_points, transformed_points)
	return image.transform(image.size, Image.PERSPECTIVE, tuple(transformation_parameters), Image.BICUBIC)


def transform_image(image: Image.Image, *, rotate: bool = True, scale: bool = True, tilt: bool = True, scale_deviation: float = 0.2, tilt_deviation: float = 0.2, generator: np.random.Generator = np.random.default_rng()) -> Image.Image:
	"""
	Apply random transformations to the image: rotation (by uniformly random angle), scale (by gaussian random factor) and tilt (by gaussian random factor).

	Note that image is cropped and converted to RGB (same channels' values in grayscale case).

	See also rotate_image, scale_image, tilt_image.

	:param image: image to apply transformations to.
	:param rotate: whether to apply rotation transformation.
	:param scale: whether to apply rotation transformation.
	:param tilt: whether to apply rotation transformation.
	:param scale_deviation: sigma parameter of normal distribution also known as standard deviation in case scale is True.
	:param tilt_deviation: maximum perspective factor.
	:param generator: random number generator.
	"""
	if rotate:
		image = rotate_image(image, generator=generator)
	if scale:
		image = scale_image(image, deviation=scale_deviation, generator=generator)
	if tilt:
		image = tilt_image(image, deviation=tilt_deviation, generator=generator)
	return image


def sar(image: Union[str, Path, Image.Image, np.ndarray], *, noise: bool = True, transform: bool = True, rotate: bool = True, scale: bool = True, tilt: bool = True, channels_weights: np.ndarray = np.array([0.299, 0.557, 0.144]), bias: float = -33, noise_mean: float = 0, noise_deviation: float = 12.5, scale_deviation: float = 0.2, tilt_deviation: float = 0.2, generator: np.random.Generator = np.random.default_rng()) -> Image.Image:
	"""
	Generate SAR image from optic one.

	Source image must contain 3 first channels: R, G, B. Resulting image contains also 3 channels: grayscale, grayscale, grayscale due to compatibility with popular image formats and processing libraries or programs.

	Generation parameters can be tuned with keyword arguments.

	:param image: text of path to image file or path object or image object or array of shape (height, width, 3) of numbers.
	:param noise:  whether to apply gaussian noise.
	:param transform: whether to apply random transformations.
	:param rotate: whether to apply rotation transformation if transform is True.
	:param scale: whether to apply rotation transformation if transform is True.
	:param tilt: whether to apply rotation transformation if transform is True.
	:param channels_weights: R, G, B coefficients array of shape 3 of floats.
	:param bias: luma bias to apply after conversion.
	:param noise_mean: mu parameter of normal distribution in case noise is True.
	:param noise_deviation: sigma parameter of normal distribution also known as standard deviation in case noise is True.
	:param scale_deviation: sigma parameter of normal distribution also known as standard deviation in case scale is True.
	:param tilt_deviation: maximum perspective factor.
	:param generator: random number generator.
	"""
	if isinstance(image, str):
		image = Path(image)
	if isinstance(image, Path):
		if not image.is_file():
			raise ValueError(f'Path {image.absolute()} does not point to an image file.')
		image = Image.open(image)
	if isinstance(image, Image.Image):
		# noinspection PyTypeChecker
		image = np.array(image)
		shape = image.shape
		if len(shape) < 3 or shape[2] < 3:
			raise ValueError(f'Image must has shape (height, width, 3) but has {shape}.')
		image = image[:, :, :3]
	if not isinstance(image, np.ndarray):
		raise ValueError('Bad data type for image.')
	image = image.astype(float)
	image = convert_image(image, channels_weights=channels_weights, bias=bias)
	if noise:
		image = noise_image(image, mean=noise_mean, deviation=noise_deviation, generator=generator)
	image = Image.fromarray(clamp_array(image, minimum=0, maximum=255).astype(np.uint8)).convert('RGB')
	if transform:
		image = transform_image(image, rotate=rotate, scale=scale, tilt=tilt, scale_deviation=scale_deviation, tilt_deviation=tilt_deviation, generator=generator)
	return image
