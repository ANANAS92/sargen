"""
Grabs map tiles from Google maps with variative parameters.
"""
import math
from enum import Enum
from io import BytesIO

from requests import get
from urllib.parse import quote
from PIL import Image


class MapMode(Enum):
	"""
	Map representation modes.
	"""
	Scheme = 'm'
	Satellite = 's'


def get_url(mode: MapMode) -> str:
	"""
	Get URL without parameters for tile fetching.
	"""
	return f'https://mt0.google.com/vt/lyrs={mode.value}'


def get_image(url: str, x: int, y: int, zoom: int, styles: str = None) -> Image:
	"""
	Get map tile as PIL image.
	"""
	url += f'?x={x}&y={y}&z={zoom}'
	if styles is not None:
		url += '&apistyle=' + styles
	response = get(url)
	if response.status_code != 200:
		raise ValueError('Can not download map tile.')
	return Image.open(BytesIO(response.content))


styleparse_types = {"all": "0", "administrative": "1", "administrative.country": "17", "administrative.land_parcel": "21", "administrative.locality": "19", "administrative.neighborhood": "20", "administrative.province": "18", "landscape": "5", "landscape.man_made": "81", "landscape.natural": "82", "poi": "2", "poi.attraction": "37", "poi.business": "33", "poi.government": "34", "poi.medical": "36", "poi.park": "40", "poi.place_of_worship": "38", "poi.school": "35", "poi.sports_complex": "39", "road": "3", "road.arterial": "50", "road.highway": "49", "road.local": "51", "transit": "4", "transit.line": "65", "transit.station": "66", "water": "6"}

styleparse_elements = {"all": "a", "geometry": "g", "geometry.fill": "g.f", "geometry.stroke": "g.s", "labels": "l", "labels.icon": "l.i", "labels.text": "l.t", "labels.text.fill": "l.t.f", "labels.text.stroke": "l.t.s"}

styleparse_stylers = {"color": "p.c", "gamma": "p.g", "hue": "p.h", "invert_lightness": "p.il", "lightness": "p.l", "saturation": "p.s", "visibility": "p.v", "weight": "p.w"}


def encode_styles(styles: list) -> str:
	"""
	Styles as JSON-like objects can be obtained from https://mapstyle.withgoogle.com or https://snazzymaps.com.
	See https://gist.github.com/sebastianleonte/617628973f88792cd097941220110233#gistcomment-3251811 and https://gist.github.com/sebastianleonte/69a5f62220fbf25dca7de86c3b6d23ac for further information.
	"""
	parsed_style = ""
	for style in styles:
		if style.get('featureType'):
			parsed_style += "s.t:" + styleparse_types[style.get("featureType")] + "|"
		if style.get('elementType'):
			parsed_style += "s.e:" + styleparse_elements[style.get("elementType")] + "|"
		for styler in style.get("stylers"):
			keys = []
			for k in styler:
				if k == "color" and len(styler[k]) == 7:
					styler[k] = "#ff" + str(styler[k][1:])
				parsed_style += styleparse_stylers[k] + ":" + str(styler[k]) + "|"
		parsed_style += ","
	return quote(parsed_style)


def get_xy(latitude: float, longitude: float, zoom: int):
	"""
	Generates an X,Y tile coordinate based on the latitude, longitude
	and zoom level
	Returns:    An X,Y tile coordinate
	See https://gist.github.com/eskriett/6038468 for further information.
	"""

	tile_size = 256

	# Use a left shift to get the power of 2
	# i.e. a zoom level of 2 will have 2^2 = 4 tiles
	numTiles = 1 << zoom

	# Find the x_point given the longitude
	point_x = (tile_size / 2 + longitude * tile_size / 360.0) * numTiles // tile_size

	# Convert the latitude to radians and take the sine
	sin_y = math.sin(latitude * (math.pi / 180.0))

	# Calulate the y coorindate
	point_y = ((tile_size / 2) + 0.5 * math.log((1 + sin_y) / (1 - sin_y)) * -(tile_size / (2 * math.pi))) * numTiles // tile_size

	return int(point_x), int(point_y)


def get_tile(latitude: float, longitude: float, zoom: int, mode: MapMode, styles: list = None) -> Image:
	if mode is MapMode.Satellite:
		styles = None
	if styles is not None:
		styles = encode_styles(styles)
	return get_image(get_url(mode), *get_xy(latitude, longitude, zoom), zoom, styles=styles)


if __name__ == '__main__':
	styles = [
		{
			"featureType": "all",
			"elementType": "all",
			"stylers": [
				{
					"color": "#ff7000"
				},
				{
					"lightness": "69"
				},
				{
					"saturation": "100"
				},
				{
					"weight": "1.17"
				},
				{
					"gamma": "2.04"
				}
			]
		},
		{
			"featureType": "all",
			"elementType": "geometry",
			"stylers": [
				{
					"color": "#cb8536"
				}
			]
		},
		{
			"featureType": "all",
			"elementType": "labels",
			"stylers": [
				{
					"color": "#ffb471"
				},
				{
					"lightness": "66"
				},
				{
					"saturation": "100"
				}
			]
		},
		{
			"featureType": "all",
			"elementType": "labels.text.fill",
			"stylers": [
				{
					"gamma": 0.01
				},
				{
					"lightness": 20
				}
			]
		},
		{
			"featureType": "all",
			"elementType": "labels.text.stroke",
			"stylers": [
				{
					"saturation": -31
				},
				{
					"lightness": -33
				},
				{
					"weight": 2
				},
				{
					"gamma": 0.8
				}
			]
		},
		{
			"featureType": "all",
			"elementType": "labels.icon",
			"stylers": [
				{
					"visibility": "off"
				}
			]
		},
		{
			"featureType": "landscape",
			"elementType": "all",
			"stylers": [
				{
					"lightness": "-8"
				},
				{
					"gamma": "0.98"
				},
				{
					"weight": "2.45"
				},
				{
					"saturation": "26"
				}
			]
		},
		{
			"featureType": "landscape",
			"elementType": "geometry",
			"stylers": [
				{
					"lightness": 30
				},
				{
					"saturation": 30
				}
			]
		},
		{
			"featureType": "poi",
			"elementType": "geometry",
			"stylers": [
				{
					"saturation": 20
				}
			]
		},
		{
			"featureType": "poi.park",
			"elementType": "geometry",
			"stylers": [
				{
					"lightness": 20
				},
				{
					"saturation": -20
				}
			]
		},
		{
			"featureType": "road",
			"elementType": "geometry",
			"stylers": [
				{
					"lightness": 10
				},
				{
					"saturation": -30
				}
			]
		},
		{
			"featureType": "road",
			"elementType": "geometry.stroke",
			"stylers": [
				{
					"saturation": 25
				},
				{
					"lightness": 25
				}
			]
		},
		{
			"featureType": "water",
			"elementType": "all",
			"stylers": [
				{
					"lightness": -20
				},
				{
					"color": "#ecc080"
				}
			]
		}
	]
	get_tile(60.061728521474635, 30.316047568204336, 18, MapMode.Satellite, styles).show()
