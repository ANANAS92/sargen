"""
Optic to SAR image converter.

Generates radio location 24 bit grayscale (LLL) images from 24 bit RGB optic images with random transformations (noise, rotation, scale,tilt).
"""
import math
from pathlib import Path
from typing import Any, Union

import numpy as np
from nptyping import NDArray, Number
from PIL import Image


def _assert_array(array: np.ndarray, signature: NDArray) -> None:
	"""
	Assert given object is array of defined shape and data type.

	Raises ValueError.

	:param array: object to check.
	:param signature: required shape and data type of array.
	"""
	try:
		if not isinstance(array, np.ndarray):
			raise
		# noinspection PyTypeChecker
		if not isinstance(array, signature):
			# noinspection PyProtectedMember,PyTypeChecker
			if not isinstance(array.astype(signature._type.npbase or signature._type.base, casting='safe'), signature):
				raise
	except Exception as exception:
		raise ValueError(f'Array must be {signature}.') from exception


def clamp_array(array: NDArray[Number], minimum: Number = 0, maximum: Number = 255) -> NDArray[Number]:
	"""
	Clamp values less then 0 to 0 and greater then 255 to 255.

	:param array: array of numbers.
	:param minimum: minimum value to clamp elements below it.
	:param maximum: maximum value to clamp elements above it.
	"""
	# noinspection PyTypeChecker
	# _assert_array(array, NDArray[Number])
	# noinspection PyTypeChecker
	return np.where(array < minimum, minimum, np.where(array > maximum, maximum, array))


def convert_image(image: NDArray[(Any, Any, 3), float], *, channels_weights: NDArray[3, float] = np.array([0.299, 0.557, 0.144]), bias: float = -33) -> NDArray[(Any, Any), float]:
	"""
	Convert R, G, B color channels to luma single channel.

	Note that luma is not clamped.

	:param image: array of pixels of WxHxCh shape.
	:param channels_weights: R, G, B coefficients.
	:param bias: luma bias to apply after conversion.
	"""
	# noinspection PyTypeChecker
	# _assert_array(image, NDArray[(Any, Any, 3), float])
	return (image * channels_weights).sum(axis=2) + bias


def noise_image(image: NDArray[(Any, Any), float], *, mean: float = 0, deviation: float = 12.5, generator: np.random.Generator = np.random.default_rng()) -> NDArray[(Any, Any), float]:
	"""
	Add gaussian noise to each pixel of single channel image.

	Note that luma is not clamped.

	:param image: array of pixels of WxHxCh shape.
	:param mean: mu parameter of normal distribution.
	:param deviation: sigma parameter of normal distribution also known as standard deviation.
	:param generator: random number generator.
	"""
	# noinspection PyTypeChecker
	# _assert_array(image, NDArray[(Any, Any), float])
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


def scale_image(image: Image.Image, *, mean: float = 1, deviation: float = 0.2, generator: np.random.Generator = np.random.default_rng()) -> Image.Image:
	"""
	Scale image by gaussian random factor.

	:param image: image to scale.
	:param mean: mu parameter of normal distribution.
	:param deviation: sigma parameter of normal distribution also known as standard deviation.
	:param generator: random number generator.
	"""
	factor = generator.normal(mean, deviation)
	if factor <= 0:
		factor = mean
	return image.resize(tuple((np.array(image.size) * factor).round(0).astype(int)))


def _find_perspective_coefficients(origin_points: NDArray[(4, 2), float], transformed_points: NDArray[(4, 2), float]) -> NDArray[8, float]:
	"""
	Find perspective transformation coefficients a, b, c, d, e, f, g, h.

	See https://stackoverflow.com/a/14178717 along with other answers and comments.

	:param origin_points: 4 origin points that are mapped to transformed.
	:param transformed_points: 4 points to where origin ones are mapped.
	"""
	# noinspection PyTypeChecker
	# _assert_array(origin_points, NDArray[(4, 2), float])
	# noinspection PyTypeChecker
	# _assert_array(transformed_points, NDArray[(4, 2), float])
	matrix_buf = []
	for src_point, dist_point in zip(origin_points, transformed_points):
		matrix_buf.append([src_point[0], src_point[1], 1, 0, 0, 0, -dist_point[0] * src_point[0], -dist_point[0] * src_point[1]])
		matrix_buf.append([0, 0, 0, src_point[0], src_point[1], 1, -dist_point[1] * src_point[0], -dist_point[1] * src_point[1]])

	a_matrix = np.array(matrix_buf, dtype=np.float)
	b_matrix = np.array(transformed_points).reshape(8)

	res = np.linalg.solve(a_matrix, b_matrix)
	return res.reshape(8)


def tilt_image(image: Image.Image, *, deviation: float = 0.2, generator: np.random.Generator = np.random.default_rng()) -> Image.Image:
	"""
	Tilt image to any direction by gaussian random factor. This is perspective transformation.

	Note that image is cropped to fit origin box.

	:param image: image to tilt and crop.
	:param deviation: maximum perspective in pixels as the smallest side multiplication factor, e.g. 0.2 from 1920x1080 results in deviation from 0 to 216 pixels in all directions.
	:param generator: random number generator.
	"""
	# See https://stackoverflow.com/a/14178717 along with other answers and comments.
	# We have 4 points = 8 coordinates that can be shifted (each in one direction only).
	# noinspection PyTypeChecker
	factors = -1 * clamp_array(generator.uniform(0, deviation * min(image.size), 8), minimum=0, maximum=deviation * min(image.size))  # invert factors to stretch image out of the clipping box
	# Shift image box points out of self by factors to avoid empty space after perspective.
	origin_points = np.array([[0, 0], [image.width, 0], [image.width, image.height], [0, image.height]])
	transformed_points = np.array([[-factors[0], -factors[1]], [image.width + factors[2], -factors[3]], [image.width + factors[4], image.height + factors[5]], [-factors[6], image.height + factors[7]]])
	# noinspection PyTypeChecker
	transformation_parameters = _find_perspective_coefficients(origin_points, transformed_points)
	return image.transform(image.size, Image.PERSPECTIVE, tuple(transformation_parameters), Image.BICUBIC)


def transform_image(image: Image.Image, *, rotate: bool = True, scale: bool = True, tilt: bool = True, scale_mean: float = 1, scale_deviation: float = 0.2, tilt_deviation: float = 0.2, generator: np.random.Generator = np.random.default_rng()) -> Image.Image:
	"""
	Apply random transformations to the image: rotation (by uniformly random angle), scale (by gaussian random factor) and tilt (by gaussian random factor).

	Note that image is cropped and converted to RGB (same channels' values in grayscale case).

	See also rotate_image, scale_image, tilt_image.

	:param image: image to apply transformations to.
	:param rotate: whether to apply rotation transformation.
	:param scale: whether to apply rotation transformation.
	:param tilt: whether to apply rotation transformation.
	:param scale_mean: mu parameter of normal distribution in case scale is True.
	:param scale_deviation: sigma parameter of normal distribution also known as standard deviation in case scale is True.
	:param tilt_deviation: maximum perspective in pixels as the smallest side multiplication factor in case tilt is True, e.g. 0.2 from 1920x1080 results in deviation from 0 to 216 pixels in all directions.
	:param generator: random number generator.
	"""
	if rotate:
		image = rotate_image(image, generator=generator)
	if scale:
		image = scale_image(image, mean=scale_mean, deviation=scale_deviation, generator=generator)
	if tilt:
		image = tilt_image(image, deviation=tilt_deviation, generator=generator)
	return image


def sar(image: Union[str, Path, Image.Image, NDArray[(Any, Any, 3), Number]], *, noise: bool = True, transform: bool = True, channels_weights: NDArray[3, float] = np.array([0.299, 0.557, 0.144]), bias: float = -33, noise_mean: float = 0, noise_deviation: float = 12.5, scale_mean: float = 1, scale_deviation: float = 0.2, tilt_deviation: float = 0.2, generator: np.random.Generator = np.random.default_rng()) -> Image.Image:
	"""
	Generate SAR image from optic one.

	Source image must contain 3 first channels: R, G, B. Resulting image contains also 3 channels: grayscale, grayscale, grayscale due to compatibility with popular image formats and processing libraries or programs.

	Generation parameters can be tuned with keyword arguments.

	:param image: text of path to image file or path object or image object or array.
	:param noise:  whether to apply gaussian noise.
	:param transform: whether to apply random transformations.
	:param channels_weights: R, G, B coefficients.
	:param bias: luma bias to apply after conversion.
	:param noise_mean: mu parameter of normal distribution in case noise is True.
	:param noise_deviation: sigma parameter of normal distribution also known as standard deviation in case noise is True.
	:param scale_mean: mu parameter of normal distribution in case scale is True.
	:param scale_deviation: sigma parameter of normal distribution also known as standard deviation in case scale is True.
	:param tilt_deviation: maximum perspective in pixels as the smallest side multiplication factor in case tilt is True, e.g. 0.2 from 1920x1080 results in deviation from 0 to 216 pixels in all directions.
	:param generator: random number generator.
	"""
	if isinstance(image, str):
		image = Path(image)
	if isinstance(image, Path):
		if not image.is_file():
			raise ValueError(f'Path {image.absolute()} does not point to an image file.')
		if not image.exists():
			raise ValueError(f'Could not find image at {image.absolute()}.')
		image = Image.open(image)
	if isinstance(image, Image.Image):
		# noinspection PyTypeChecker
		image = np.array(image)
		shape = image.shape
		if len(shape) < 3 or shape[2] < 3:
			raise ValueError(f'Image must has shape (..., ..., 3) but has {shape}.')
		image = image[:, :, :3]
	image = image.astype(float)
	# noinspection PyTypeChecker
	image = convert_image(image, channels_weights=channels_weights, bias=bias)
	if noise:
		image = noise_image(image, mean=noise_mean, deviation=noise_deviation, generator=generator)
	image = Image.fromarray(clamp_array(image, minimum=0, maximum=255).astype(np.uint8)).convert('RGB')
	if transform:
		image = transform_image(image, scale_mean=scale_mean, scale_deviation=scale_deviation, tilt_deviation=tilt_deviation, generator=generator)
	return image
