# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:25:01 2021

@author: DELL
"""

import json, os, pyproj, urllib.request, cv2
import numpy as np
from PIL import Image
from io import BytesIO
# from pyproj import Proj, transform
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from shapely.geometry import Point, LineString, Polygon
from matplotlib.pyplot import imread
from pyproj import Transformer
import copy, shapely
import shutil

geod = pyproj.Geod(ellps='WGS84')


def valid_lonlat(lon, lat):
	# Put the longitude in the range of [0,360):
	lon %= 360
	# Put the longitude in the range of [-180,180):
	if lon >= 180:
		lon -= 360
	lon_lat_point = shapely.geometry.Point(lon, lat)
	lon_lat_bounds = shapely.geometry.Polygon.from_bounds(
		xmin=-180.0, ymin=-90.0, xmax=180.0, ymax=90.0
	)
	# return lon_lat_bounds.intersects(lon_lat_point)
	# would not provide any corrected values

	if lon_lat_bounds.intersects(lon_lat_point):
		return lon, lat


def dist_n(p0, p1):
	a, b = Point(p0), Point(p1)
	angle1, angle2, distance1 = geod.inv(a.x, a.y, b.x, b.y)
	return distance1


def plot_line(ax, ob, color, w=0.5, alpha=1):
	x, y = ob.xy
	ax.plot(x, y, color=color, linewidth=w, alpha=1, solid_capstyle='round', zorder=1)


def formart_point(point):
	return (float("{:.5f}".format(point[0])), float("{:.5f}".format(point[1])))


def makeRequest(rq):
	f = urllib.request.urlopen(rq)
	data = f.read()
	return data


def find_folder(direction_out):
	if os.path.isdir(direction_out):
		shutil.rmtree(direction_out)
	os.mkdir(direction_out)


#    except:
#        pass


def new_directions(direct, name):
	directions = {}
	for key in direct.keys():
		find_folder(name + '\\' + key)
		for d in direct[key].keys():
			find_folder(name + '\\' + key + '\\' + direct[key][d])
			directions[d] = name + '\\' + key + '\\' + direct[key][d]
	return directions


def set_points(start, finish, step_length):
	dist = dist_n(start, finish)
	step = step_length
	view = step_length
	if dist % step_length == 0:
		delta = int(dist // step) + 1
	else:
		delta = int(dist // step) + 2
	return dist, step, view, delta


def get_list_points(start, finish, delta, sis_coord):
	transformer = Transformer.from_crs(sis_coord['WGS_84'], sis_coord['Pseudo_Mercator'])
	transformer_84 = Transformer.from_crs(sis_coord['Pseudo_Mercator'], sis_coord['WGS_84'])
	l = []
	point0 = transformer.transform(start[1], start[0])
	point1 = transformer.transform(finish[1], finish[0])
	xquery = np.linspace(point0[0], point1[0], delta)
	yquery = np.linspace(point0[1], point1[1], delta)
	for j in range(len(xquery)):
		p = transformer_84.transform(xquery[j], yquery[j])
		new_p = formart_point(p)
		l.append((new_p[1], new_p[0]))
	return l


def extend_path(start, finish, step_length, remain, dist, sis_coord):
	transformer = Transformer.from_crs(sis_coord['WGS_84'], sis_coord['Pseudo_Mercator'])
	transformer_84 = Transformer.from_crs(sis_coord['Pseudo_Mercator'], sis_coord['WGS_84'])
	dist_to_new_point = int(step_length - remain) + 1
	point0 = transformer.transform(start[1], start[0])
	point1 = transformer.transform(finish[1], finish[0])
	detX = (point1[0] - point0[0]) / dist
	delY = (point1[1] - point0[1]) / dist
	new_finish = (point1[0] + (dist_to_new_point * detX), point1[1] + (dist_to_new_point * delY))
	n_finish = transformer_84.transform(new_finish[0], new_finish[1])
	return (n_finish[1], n_finish[0])


def list_of_points(path, direct, step_length, shooting_radius, sis_coord):
	if len(path) == 2:
		start, finish = formart_point(path[0]), formart_point(path[1])
		name = str(path[0]) + '_' + str(path[1])
		directions = new_directions(direct, name)
		dist, step, view, delta = set_points(start, finish, step_length, shooting_radius)
		remain = dist % step_length
		if remain != 0:
			new_finish = extend_path(start, finish, step_length, remain, dist, sis_coord)
			dist, step, view, delta = set_points(start, new_finish, step_length, shooting_radius)
			Point_WGS84 = get_list_points(start, new_finish, delta, sis_coord)
		else:
			Point_WGS84 = get_list_points(start, finish, delta, sis_coord)
	return directions, view, Point_WGS84, name


def latlon_data(start, finish, binKey, type_of_imagery):
	lat_max, lat_min = max([start[1], finish[1]]), min([start[1], finish[1]])
	lon_max, lon_min = max([start[0], finish[0]]), min([start[0], finish[0]])
	rq = "http://dev.virtualearth.net/REST/V1/Imagery/Map/{0}?mapArea={1},{2},{3},{4}&zoomLevel=19&ms=1000,100&mmd=0&key={5}"
	rq = "http://dev.virtualearth.net/REST/V1/Imagery/Map/{0}?mapArea={1},{2},{3},{4}&zoomLevel=19&ms=1000,1000&mmd={5}&key={6}"
	rq1 = rq.format(type_of_imagery, lat_min, lon_min, lat_max, lon_max, 0, binKey)
	rq2 = rq.format(type_of_imagery, lat_min, lon_min, lat_max, lon_max, 1, binKey)
	return rq1, rq2


def get_bing_tile(start, finish, binKey, type_of_imagery):
	rq1, rq2 = latlon_data(start, finish, binKey, type_of_imagery)
	tile = makeRequest(rq1)
	retJson = makeRequest(rq2).decode('utf-8')
	obj = json.loads(retJson)
	zoom = int(obj['resourceSets'][0]['resources'][0]['zoom'])
	#    print('Zoom: ',zoom)
	coords = obj['resourceSets'][0]['resources'][0]['bbox']
	size_picture = (int(obj['resourceSets'][0]['resources'][0]['imageWidth']), int(obj['resourceSets'][0]['resources'][0]['imageHeight']))
	return [coords[1], coords[3], coords[0], coords[2]], size_picture, zoom, tile


def get_tile_dict(Point_WGS84, binKey, type_of_imagery, name, direction_output):
	Tiles, Z = {}, {}
	for i in range(len(Point_WGS84) - 1):
		coords, size_picture, zoom, tile = get_bing_tile(Point_WGS84[i], Point_WGS84[i + 1], binKey, type_of_imagery)
		Tiles[(Point_WGS84[i], Point_WGS84[i + 1])] = {'box': [(coords[0], coords[2]), (coords[0], coords[3]), (coords[1], coords[3]), (coords[1], coords[2])],
		                                               'size': size_picture,
		                                               'del_width': dist_n((coords[0], coords[2]), (coords[1], coords[2])) / size_picture[0],
		                                               'del_high': dist_n((coords[0], coords[2]), (coords[0], coords[3])) / size_picture[1],
		                                               'polygon': Polygon([(coords[0], coords[2]), (coords[0], coords[3]), (coords[1], coords[3]), (coords[1], coords[2]), (coords[0], coords[2])]),
		                                               'line': LineString([Point_WGS84[i], Point_WGS84[i + 1]]),
		                                               'start': Point_WGS84[i],
		                                               'finish': Point_WGS84[i + 1],
		                                               'zoom': zoom,
		                                               'tile': tile,
		                                               'id': i}
		Z[(Point_WGS84[i], Point_WGS84[i + 1])] = zoom
		image = Image.open(BytesIO(tile))
		try:
			path = os.path.join(direction_output, 'tile_' + name + '_' + str(i) + '.jpg')
			image.save(path)
		except:
			path = os.path.join(direction_output, 'tile_' + name + '_' + str(i) + '.png')
			image.save(path)

		Tiles[(Point_WGS84[i], Point_WGS84[i + 1])]['name_pict'] = name + '_' + str(i)
		image.close()
	return Tiles, Z


def path_plan(Tiles, Point_WGS84, ax):
	L = []
	for i in range(len(Point_WGS84) - 1):
		points = (Point_WGS84[i], Point_WGS84[i + 1])
		if points in Tiles.keys():
			L.append(points)
			plot_line(ax, Tiles[points]['line'], 'red', w=1, alpha=1)
			ax.plot(Tiles[points]['start'][0], Tiles[points]['start'][1], 'o', markersize=7, color='red')
			ax.plot(Tiles[points]['finish'][0], Tiles[points]['finish'][1], 'o', markersize=6, color='blue')
			ax.add_patch(PolygonPatch(Tiles[points]['polygon'], fc='gray', ec='gray', alpha=0.2, zorder=1))
		else:
			raise Exception('error', points)
	ax.set_aspect(1)
	return L, ax


def get_parametrs(start, finish, first_tile, Tiles, L):
	if start[0] < finish[0]:  # х увеличивается
		width = 0  # ширина рисунка начинается с 0
		if start[1] < finish[1]:  # y увеличивается
			high = Tiles[first_tile]['size'][1] * (len(L) - 1)  # высота рисунка начинается с последнего тайла
		else:
			high = 0
	elif start[0] > finish[0]:
		width = Tiles[L[0]]['size'][0] * (len(L) - 1)  # ширина рисунка начинается с последнего тайла
		if start[1] < finish[1]:  # y увеличивается
			high = Tiles[first_tile]['size'][1] * (len(L) - 1)  # высота рисунка начинается с последнего тайла
		else:
			high = 0  # высота рисунка начинается с 0
	else:
		width = 0  # ширина рисунка начинается с 0
		if start[1] < finish[1]:  # y увеличивается
			high = Tiles[first_tile]['size'][1] * (len(L) - 1)  # высота рисунка начинается с последнего тайла
		else:
			high = 0
	return width, high


def size_pict(W, H, C):
	C_new = copy.deepcopy(C)
	deltaY, deltaX = min(H), min(W)
	cofY = -1
	cofX = -1
	for c in C_new.keys():
		C_new[c] = (C[c][0] + cofX * deltaX, C[c][1] + cofY * deltaY)
	sizeH = max(H) + cofY * deltaY
	sizeW = max(W) + cofX * deltaX
	return sizeW, sizeH, C_new


def merge_pict(Tiles, L, Point_WGS84, directions, name, ax):
	first_tile = L[0]
	width, high = get_parametrs(Tiles[first_tile]['start'], Tiles[first_tile]['finish'], first_tile, Tiles, L)
	W, H = [width], [high]
	C = {first_tile: (width, high)}

	for i in range(1, len(L)):
		key, priv_key = L[i], L[i - 1]

		intersect_poly = Tiles[priv_key]['polygon'].intersection(Tiles[key]['polygon'])
		ax.add_patch(PolygonPatch(intersect_poly, fc='blue', ec='blue', alpha=0.2, zorder=1))
		bounds = intersect_poly.bounds

		for idy in [1, 3]:
			point = (bounds[0], bounds[idy])
			if Point(point).touches(Tiles[priv_key]['polygon']) and Point(point).touches(Tiles[key]['polygon']):
				ax.plot(point[0], point[1], 'o', markersize=10, color='green')
				intersect_point = point
				idY = idy

		if Tiles[priv_key]['box'][0][0] - Tiles[key]['box'][0][0] > 0:  # клетка находится левеее(х уменьшается)
			if idY == 3:
				id_box = 1
			else:
				id_box = 0
			plot_line(ax, LineString([intersect_point, Tiles[key]['box'][id_box]]), 'darkgreen', w=3, alpha=1)
			del_width = dist_n(intersect_point, Tiles[key]['box'][id_box])
			width = C[priv_key][0] - round(del_width / Tiles[key]['del_width'])

			if Tiles[priv_key]['box'][0][1] - Tiles[key]['box'][0][1] > 0:  # клетка находится ниже  (у уменьшается)
				plot_line(ax, LineString([intersect_point, Tiles[priv_key]['box'][id_box]]), 'orange', w=3.5, alpha=1)
				del_high = dist_n(intersect_point, Tiles[priv_key]['box'][id_box])
				high = C[priv_key][1] + round(del_high / Tiles[priv_key]['del_high'])
			else:  # клетка находится выше  (у увеличивается)
				plot_line(ax, LineString([intersect_point, Tiles[priv_key]['box'][id_box]]), 'orange', w=3.5, alpha=1)
				del_high = dist_n(intersect_point, Tiles[priv_key]['box'][id_box])
				high = C[priv_key][1] - round(del_high / Tiles[priv_key]['del_high'])

		elif Tiles[priv_key]['box'][0][0] - Tiles[key]['box'][0][0] < 0:  # клетка находится правее(х увеличиваетсся)
			if idY == 3:
				id_box = 1
			else:
				id_box = 0
			plot_line(ax, LineString([intersect_point, Tiles[priv_key]['box'][id_box]]), 'darkgreen', w=3, alpha=1)
			del_width = dist_n(intersect_point, Tiles[priv_key]['box'][id_box])
			width = C[priv_key][0] + round(del_width / Tiles[priv_key]['del_width'])

			if Tiles[priv_key]['box'][0][1] - Tiles[key]['box'][0][1] > 0:  # клетка находится ниже  (у уменьшается)
				plot_line(ax, LineString([intersect_point, Tiles[key]['box'][id_box]]), 'orange', w=3.5, alpha=1)
				del_high = dist_n(intersect_point, Tiles[key]['box'][id_box])
				high = C[priv_key][1] + round(del_high / Tiles[key]['del_high'])

			else:  # клетка находится выше  (у увеличивается)
				plot_line(ax, LineString([intersect_point, Tiles[key]['box'][id_box]]), 'orange', w=3.5, alpha=1)
				del_high = dist_n(intersect_point, Tiles[key]['box'][id_box])
				high = C[priv_key][1] - round(del_high / Tiles[key]['del_high'])
		else:
			width = C[priv_key][0]
			ax.plot(bounds[0], bounds[3], 'o', markersize=10, color='orange')
			p0, p1 = (bounds[0], bounds[1]), (bounds[0], bounds[3])
			plot_line(ax, LineString([p0, p1]), 'orange', w=2, alpha=1)
			del_high = dist_n(p0, p1)
			if Tiles[priv_key]['box'][0][1] - Tiles[key]['box'][0][1] > 0:  # клетка находится ниже  (у уменьшается)
				high = C[priv_key][1] + Tiles[priv_key]['size'][0] - round(del_high / Tiles[priv_key]['del_high'])
			else:
				high = C[priv_key][1] - Tiles[priv_key]['size'][0] + round(del_high / Tiles[priv_key]['del_high'])

		C[key] = (width, high)
		W.append(width)
		H.append(high)

	plt.savefig(str(name) + '/plan.jpg')
	plt.close()
	return W, H, C


def final_pict(W, H, C, Tiles, direction_tile, direction, name, num0, num1):
	first_tile = list(Tiles.keys())[0]
	size_pictures = Tiles[first_tile]['size']
	new_image = Image.new('RGBA', (W + size_pictures[0], H + size_pictures[1]), (250, 250, 250))
	i = 0
	for key in C.keys():
		i += 1
		name_tile = Tiles[key]['name_pict']
		try:
			path = os.path.join(direction_tile, 'tile_' + name_tile + '.jpg')
			image = Image.open(path, 'r')
		except:
			path = os.path.join(direction_tile, 'tile_' + name_tile + '.png')
			image = Image.open(path, 'r')
		new_image.paste(image, C[key])
	rgb_im = new_image.convert('RGBA')
	path1 = os.path.join(direction, 'merged_' + name + '_' + str(num0) + '_' + str(num1) + '.png')
	rgb_im.save(path1)
	del new_image


def save_merged_image(sizeW, sizeH, C_new, L, Tiles, name, directions):
	first_tile = list(Tiles.keys())[0]
	size_pictures = Tiles[first_tile]['size']
	Merged_tiles = {}
	if len(L) > 10:
		List_keys = list(C_new.keys())
		keys = []
		if len(List_keys) // 10 == 1:
			num_pict = len(List_keys) // 2
			keys = [(0, num_pict + 1), (num_pict, len(List_keys) - 1)]

		else:
			if len(List_keys) % 10 > 5:
				num_pict = len(List_keys) // 10 + 1
				tiles_in_pict = int(len(List_keys) // num_pict)
			else:
				num_pict = len(List_keys) // 10
				tiles_in_pict = int(len(List_keys) // num_pict)

			for j in range(num_pict - 1):
				keys.append((j * tiles_in_pict, (j + 1) * (tiles_in_pict) + 1))
			keys.append(((num_pict - 1) * tiles_in_pict, len(List_keys) - 1))
		id_pict = 0
		for pair in keys:
			Merged_tiles[str(pair[0]) + '_' + str(pair[1])] = {'id_tiles': []}
			list0 = list(List_keys[pair[0]:pair[1]])
			C_ten, W_ten, H_ten = {}, [], []
			X, Y = [], []
			num = 0
			for l in list0:
				Merged_tiles[str(pair[0]) + '_' + str(pair[1])]['id_tiles'].append(l)
				for point in Tiles[l]['box']:
					X.append(point[0])
					Y.append(point[1])
				C_ten[l] = C_new[l]
				W_ten.append(C_new[l][0])
				H_ten.append(C_new[l][1])
				sizeW3, sizeH3, C_new3 = size_pict(W_ten, H_ten, C_ten)
				final_pict(sizeW3, sizeH3, C_new3, Tiles, directions['out_tiles'], directions['step_by_step'], name, id_pict, num)
				num += 1
			id_pict += 1

			sizeW2, sizeH2, C_new2 = size_pict(W_ten, H_ten, C_ten)
			final_pict(sizeW2, sizeH2, C_new2, Tiles, directions['out_tiles'], directions['merged_tiles'], name, pair[0], pair[1])
			Merged_tiles[str(pair[0]) + '_' + str(pair[1])]['X'] = (min(X), max(X))
			Merged_tiles[str(pair[0]) + '_' + str(pair[1])]['Y'] = (min(Y), max(Y))
			Merged_tiles[str(pair[0]) + '_' + str(pair[1])]['size_img'] = (sizeW2 + size_pictures[0], sizeH2 + size_pictures[1])

	else:
		Merged_tiles[str(0) + '_' + str(len(L))] = {'id_tiles': []}
		C_ten, W_ten, H_ten = {}, [], []
		X, Y = [], []
		num = 0
		for c in C_new.keys():
			Merged_tiles[str(0) + '_' + str(len(L))]['id_tiles'].append(c)
			for point in Tiles[c]['box']:
				X.append(point[0])
				Y.append(point[1])
			C_ten[c] = C_new[c]
			W_ten.append(C_new[c][0])
			H_ten.append(C_new[c][1])
			sizeW3, sizeH3, C_new3 = size_pict(W_ten, H_ten, C_ten)
			final_pict(sizeW3, sizeH3, C_new3, Tiles, directions['out_tiles'], directions['step_by_step'], name, 0, num)
			num += 1

		Merged_tiles[str(0) + '_' + str(len(L))]['X'] = (min(X), max(X))
		Merged_tiles[str(0) + '_' + str(len(L))]['Y'] = (min(Y), max(Y))
		Merged_tiles[str(0) + '_' + str(len(L))]['size_img'] = (sizeW + size_pictures[0], sizeH + size_pictures[1])
		final_pict(sizeW, sizeH, C_new, Tiles, directions['out_tiles'], directions['merged_tiles'], name, 0, len(L))
	return Merged_tiles


def get_angle(point0, point1):
	P1 = Point(point1)
	P2 = Point((point0[0], point1[1]))
	interP = Point(point0)

	dx = P1.x - interP.x
	dy = P1.y - interP.y

	dx2 = interP.x - P2.x
	dy2 = interP.y - P2.y

	azimuth1 = np.arctan2(dx, dy) * 180 / np.pi
	azimuth2 = np.arctan2(dx2, dy2) * 180 / np.pi

	azimuth11 = 360 - azimuth1  # dx>0 & dy<0
	azimuth22 = 180 - azimuth2  # dx<0 & dy>0
	angle = -(azimuth11 - azimuth22)
	return angle


def get_mask(size_image, key, l, Merged_tiles, name, new_name, id_i, shooting_radius, direction_output_mask, Data_json, sis_coord):
	transformer = Transformer.from_crs(sis_coord['WGS_84'], sis_coord['Pseudo_Mercator'])
	transformer_84 = Transformer.from_crs(sis_coord['Pseudo_Mercator'], sis_coord['WGS_84'])
	plt.rcParams['savefig.facecolor'] = 'black'
	fig = plt.figure(figsize=(size_image[0], size_image[1]), dpi=1)  # frameon=False,
	fig.patch.set_facecolor('black')
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	ax.patch.set_facecolor('black')
	fig.add_axes(ax)
	point0 = transformer.transform(l[0][1], l[0][0])
	point1 = transformer.transform(l[1][1], l[1][0])

	angle = get_angle(point0, point1)

	minXY = (min(Merged_tiles[key]['X']), min(Merged_tiles[key]['Y']))
	maxXY = (max(Merged_tiles[key]['X']), max(Merged_tiles[key]['Y']))

	min_point = transformer.transform(minXY[1], minXY[0])
	max_point = transformer.transform(maxXY[1], maxXY[0])
	p1, p3 = transformer.transform(maxXY[1], minXY[0]), transformer.transform(minXY[1], maxXY[0])

	poly = Polygon([(min_point[0], min_point[1]), (p1[0], p1[1]), (max_point[0], max_point[1]), (p3[0], p3[1])])
	ax.add_patch(PolygonPatch(poly, fc='black', ec='black', alpha=1, lw=100))
	line = LineString([point0, point1])
	dilated = line.buffer(shooting_radius, cap_style=2, join_style=2)
	bound = dilated.bounds
	a1 = transformer_84.transform(bound[0], bound[1])
	a2 = transformer_84.transform(bound[2], bound[3])
	Data_json[str(key) + '_' + str(id_i)] = {'start': (l[0][1], l[0][0]), 'finish': (l[1][1], l[1][0]), 'minXY_box:': a1, 'maxXY_box:': a2}

	ax.add_patch(PolygonPatch(dilated, fc='white', ec='white', alpha=1, zorder=1))

	xrange, yrange = [min_point[0], max_point[0]], [min_point[1], max_point[1]]
	ax.set_xlim(*xrange)
	ax.set_ylim(*yrange)
	ax.set_aspect(1)
	path_buf = os.path.join(direction_output_mask, 'mask_' + new_name + '_' + str(key) + '_' + str(id_i) + '.jpg')

	plt.savefig(path_buf, facecolor='black', transparent=False)
	plt.close()
	del fig

	return angle, Data_json


def crop_image(direction_merged, direction_output_mask, direction_final, name, new_name, key, id_i):
	path_merged = os.path.join(direction_merged, 'merged_' + name + '_' + str(key) + '.png')
	mTile = Image.open(path_merged, 'r')
	map_tile = np.asarray(mTile)
	mTile.close()

	path_buf = os.path.join(direction_output_mask, 'mask_' + new_name + '_' + str(key) + '_' + str(id_i) + '.jpg')
	buffer = imread(path_buf, cv2.IMREAD_GRAYSCALE)
	gray = cv2.cvtColor(buffer, cv2.COLOR_BGR2GRAY)
	_, mask = cv2.threshold(gray, thresh=180, maxval=255, type=cv2.THRESH_BINARY)

	tile_x, tile_y, _ = map_tile.shape
	buffer_x, buffer_y = mask.shape

	x_buffer = min(tile_x, buffer_x)
	x_half_buffer = mask.shape[0] // 2
	buffer_mask = mask[x_half_buffer - x_buffer // 2: x_half_buffer + x_buffer // 2 + 1, :tile_y]
	tile_to_mask = map_tile[x_half_buffer - x_buffer // 2: x_half_buffer + x_buffer // 2 + 1, :tile_y]

	masked = cv2.bitwise_and(tile_to_mask, tile_to_mask, mask=buffer_mask)
	tmp = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
	_, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
	b, g, r, a = cv2.split(masked)
	rgba = [b, g, r, alpha]
	masked_tr = cv2.merge(rgba, 4)

	img = Image.fromarray(masked_tr, 'RGBA')
	img.save(direction_final + '/not_rotated_' + new_name + '_' + str(key) + '_' + str(id_i) + '.png')
	img.close()

	image = cv2.imread(direction_final + '/not_rotated_' + new_name + '_' + str(key) + '_' + str(id_i) + '.png')
	original_image = image
	gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray_img, 50, 255)
	contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	x, y = [], []
	for contour_line in contours:
		for contour in contour_line:
			x.append(contour[0][0])
			y.append(contour[0][1])
	x1, x2, y1, y2 = min(x), max(x), min(y), max(y)
	cropped = original_image[y1:y2, x1:x2]
	cv2.imwrite(direction_final + '/crop_' + new_name + '_' + str(key) + '_' + str(id_i) + '.png', cropped)


def rotate_img(angle, direction_final, direction_rotated, name, new_name, key, id_i):
	path_final = os.path.join(direction_final, 'crop_' + new_name + '_' + str(key) + '_' + str(id_i) + '.png')
	final_img = Image.open(path_final, 'r')
	map_image = np.asarray(final_img)
	rotation_matrix = cv2.getRotationMatrix2D((map_image.shape[0] / 2, map_image.shape[1] / 2), angle, 1)
	rotated_image = cv2.warpAffine(map_image, rotation_matrix, (map_image.shape[0], map_image.shape[1]), flags=cv2.INTER_LINEAR)

	img = Image.fromarray(rotated_image, 'RGB')
	img.save(direction_rotated + '/' + new_name + '.png')
	del img


def function_call(type_img, path, crop_step, binKey):
	sis_coord = {'Pseudo_Mercator': 3857, 'WGS_84': 4326}
	direct = {'output': {'out_masks': 'masks', 'out_tiles': 'tiles', 'out_not_rotated': 'not_rotated'},
	          'final': {'merged_tiles': 'merge_tiles', 'rotated_pict': 'rotated_pict', 'step_by_step': 'merging_of_tiles'}}
	Type_of_imagery = {'Satellite': 'Aerial',
	                   'SatelliteLabels': 'AerialWithLabels',
	                   'Road': 'Road',
	                   'DarkRoad': 'CanvasDark',
	                   'LightRoad': 'CanvasLight',
	                   'GrayRoad': 'CanvasGray'}

	step_length = 100
	start = (formart_point(path[0])[1], formart_point(path[0])[0])
	finish = (formart_point(path[1])[1], formart_point(path[1])[0])

	if dist_n(start, finish) < crop_step:
		raise ValueError("not valid parameter 'crop_step'")

	lon_lat_s = valid_lonlat(start[1], start[0])
	lon_lat_f = valid_lonlat(finish[1], finish[0])
	if lon_lat_s is None and lon_lat_f is None:
		raise ValueError("(lon, lat) is not in WGS84 bounds")
	else:
		start = (lon_lat_s[1], lon_lat_s[0])
		finish = (lon_lat_f[1], lon_lat_f[0])

		name = type_img + '_' + str(path[0]) + '_' + str(path[1])
		find_folder(name)
		directions = new_directions(direct, name)
		dist, step, view, delta = set_points(start, finish, step_length)

		remain = dist % step_length
		if remain != 0:
			new_finish = extend_path(start, finish, step_length, remain, dist, sis_coord)
			Point_WGS84 = get_list_points(start, new_finish, delta, sis_coord)
		else:
			Point_WGS84 = get_list_points(start, finish, delta, sis_coord)

		Tiles, Z = get_tile_dict(Point_WGS84, binKey, Type_of_imagery[type_img], name, directions['out_tiles'])
		# a1ll_zooms=dict(zip(list(Z.values()),[list(Z.values()).count(i) for i in list(Z.values())]))

		fig, ax = plt.subplots(num=None, figsize=(24, 12), dpi=100)
		L, ax = path_plan(Tiles, Point_WGS84, ax)

		W, H, C = merge_pict(Tiles, L, Point_WGS84, directions, name, ax)
		sizeW, sizeH, C_new = size_pict(W, H, C)
		Merged_tiles = save_merged_image(sizeW, sizeH, C_new, L, Tiles, name, directions)
		Data_json = {}
		Lines = []
		for key in Merged_tiles.keys():
			path_tile = os.path.join(directions['merged_tiles'], 'merged_' + name + '_' + str(key) + '.png')
			mTile = Image.open(path_tile, 'r')
			size_image = mTile.size
			mTile.close()
			L = Merged_tiles[key]['id_tiles']

			s, f = L[0][0], L[-1][1]
			dist, step, view, delta = set_points(s, f, crop_step)
			Point_WGS84 = get_list_points(s, f, delta - 1, sis_coord)
			id_i = 0
			for i in range(len(Point_WGS84) - 1):
				l = (Point_WGS84[i], Point_WGS84[i + 1])
				new_name = str([Point_WGS84[i][1], Point_WGS84[i][0]]) + ',' + str([Point_WGS84[i + 1][1], Point_WGS84[i + 1][0]])
				if l not in Lines:
					Lines.append(l)
					angle, Data_json = get_mask(size_image, key, l, Merged_tiles, name, new_name, id_i, crop_step - 20, directions['out_masks'], Data_json, sis_coord)
					crop_image(directions['merged_tiles'], directions['out_masks'], directions['out_not_rotated'], name, new_name, key, id_i)
					rotate_img(angle, directions['out_not_rotated'], directions['rotated_pict'], name, new_name, key, id_i)
				id_i += 1
		with open(name + '\coordinates.json', 'w') as f:
			json.dump(Data_json, f, ensure_ascii=False, indent=4)


###########TESTS#######


def test1(type_img, crop_step, binKey):
	path = [[40.754399, -73.98669], [40.769008, -73.97391]]
	function_call(type_img, path, crop_step, binKey)


def test2(type_img, crop_step, binKey):
	path = [[38.708838, -9.131419], [38.712479, -9.139875]]
	function_call(type_img, path, crop_step, binKey)


def test3(type_img, crop_step, binKey):
	sis_coord = {'Pseudo_Mercator': 3857, 'WGS_84': 4326}
	transformer_84_PM = Transformer.from_crs(sis_coord['WGS_84'], sis_coord['Pseudo_Mercator'])
	transformer_PM_84 = Transformer.from_crs(sis_coord['Pseudo_Mercator'], sis_coord['WGS_84'])
	###создание маршрута
	start = [47.273392, 39.654624]
	start_PM = transformer_84_PM.transform(start[0], start[1])
	finish_PM = [start_PM[0] - 690, start_PM[1] - 700]
	finish = transformer_PM_84.transform(finish_PM[0], finish_PM[1])
	########
	path = [start, finish]
	function_call(type_img, path, crop_step, binKey)


def test4(type_img, crop_step, binKey):
	path = [[59.949893, 30.314885], [59.951881, 30.308791]]
	function_call(type_img, path, crop_step, binKey)


#############

if __name__ == '__main__':
	###НА ВХОДЕ: 1. type_img - тип скачиваемых тайлов type_of_imagery
	###          2. маршрут в виде списка 2 точки [start, finish];
	###             point = [широта,долгота] - точки задаются листом ;
	###          3. crop_step - Дискретный шаг для вырезания картинки, задается в метрах,
	###          4. Ключ BingAPI.
	###

	###НА ВЫХОДЕ:
	###         КАТАЛОГ c названием, определяющим тип карты, первую и последнюю точку маршрута, который содержит след каталоги и файлы:
	###						1. '/output' - каталог с вспомогательными картинками, которые получаются походу работы кода
	###									* '\tiles' - скачанные тайлы
	###									* '\masks' - созданные маски для вырезания кадров
	###									* '\not_rotated' - вырезанные кадры, не повернутые 
	###						
	###						2. '/final' - каталог, где содержатся финальные картинки
	###									* '\merge_tiles' - объединенные тайлы, если расстояние между точками start, finish > 1000, разбиваются на несколько наборов, чтобы картинки были приемлемого размера
	###									* '\merging_of_tiles' - картинки с постепенным объединением тайлов
	###									* '\rotated_pict' - набор финальных вертикальных картинок 
	###						3.  'coordinates.json' - данные о финальных картинках, старт, финиш, левая нижняя и правая верхняя точки вырезанного кадра   
	###                     4.  'plan(___).jpeg' - схематичное представление всей съемки

	binKey = "AjJhQyVMzBNnY6-64Wt0GpVT_MckgYdZYCP5tSOS4mAkhjY1Pso5FEiGN9nNf4et"
	type_of_imagery = ['Satellite', 'SatelliteLabels', 'Road', 'DarkRoad', 'LightRoad', 'GrayRoad']

	crop_step = 100
	test1('Satellite', crop_step, binKey)
	test2('Satellite', crop_step, binKey)
	test3('Satellite', crop_step, binKey)
	test4('Satellite', crop_step, binKey)
	#
	test1('Road', crop_step, binKey)
	test2('DarkRoad', crop_step, binKey)
	test3('LightRoad', crop_step, binKey)
	test4('GrayRoad', crop_step, binKey)
