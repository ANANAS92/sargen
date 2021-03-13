"""
Simulate UAV fly.
"""
from __future__ import annotations
import struct
from typing import List, Optional, Union

import numpy as np
from geographiclib.geodesic import Geodesic


def move_forward(start_point: np.ndarray, direction: np.ndarray, step_velocity: float, steps: float) -> np.ndarray:
	return start_point + direction * step_velocity * steps


def normalize_vector(from_point: np.ndarray, to_point: np.ndarray) -> np.ndarray:
	vector = to_point - from_point
	length = np.linalg.norm(vector)
	direction = vector / length
	return direction


def linear_interpole(from_point: np.ndarray, to_point: np.ndarray, step_velocity: float) -> List[np.ndarray]:
	vector = to_point - from_point
	length = np.linalg.norm(vector)
	direction = normalize_vector(from_point, to_point)
	points = [from_point]
	for _ in range(int(length // step_velocity)):
		last_point = points[-1]
		points.append(move_forward(last_point, direction, step_velocity, 1))
	return points


def linear_hop(last_point: Optional[np.ndarray], from_point: np.ndarray, to_point: np.ndarray, step_velocity: float) -> np.ndarray:
	if last_point is None:
		return from_point
	direction = normalize_vector(from_point, to_point)
	if np.isclose(last_point, from_point).all():
		return move_forward(from_point, direction, step_velocity, 1)
	else:
		return from_point + direction * (step_velocity - np.linalg.norm(from_point - last_point))


def interpolate(path: List[np.ndarray], step_velocity: float) -> List[np.ndarray]:
	points = [None]
	for i in range(1, len(path)):
		points += linear_interpole(linear_hop(points[-1], path[i - 1], path[i], step_velocity), path[i], step_velocity)
	return points[1:]


def karney_distance(point1: np.ndarray, point2: np.ndarray) -> float:
	"""
	Get distance in meters between geodesic points (WGS84).
	"""
	return Geodesic.WGS84.Inverse(point1[0], point1[1], point2[0], point2[1], Geodesic.DISTANCE)['s12']


def get_local_coordinates(point: np.ndarray, reference_point: np.ndarray) -> np.ndarray:
	"""
	Get local (x,y) coordinates from geodesic coordinates and reference (zero) point geodesic coordinates (WGS84).
	"""
	return np.array((karney_distance(reference_point, np.array((reference_point[0], point[1]))), karney_distance(reference_point, np.array((point[0], reference_point[1])))))


if __name__ == '__main__':
	P = lambda x, y: np.array((x, y))
	print(interpolate([P(0, 0), P(0, 1), P(1, 1), P(0, 1)], 0.1))
	assert karney_distance(P(48.8534, 2.3488), P(51.5085, -0.12574)) - 344e3 < 1e3  # Distance between Paris and London is 344 km (assume tolerance is 1 km)
	print('Error is', abs(karney_distance(P(48.8534, 2.3488), P(51.5085, -0.12574)) - np.linalg.norm(get_local_coordinates(P(48.8534, 2.3488), P(51.5085, -0.12574)))))
