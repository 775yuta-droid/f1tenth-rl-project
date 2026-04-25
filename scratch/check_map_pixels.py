import cv2
import numpy as np
import yaml

map_path = '/workspace/my_maps/testmap-tamoku/map-tamoku.pgm'
yaml_path = '/workspace/my_maps/testmap-tamoku/map-tamoku.yaml'

with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
height, width = img.shape
origin = config['origin']
resolution = config['resolution']

def get_pixel(x, y):
    px = int((x - origin[0]) / resolution)
    py_unflipped = int((y - origin[1]) / resolution)
    py_flipped = height - py_unflipped
    
    val_unflipped = img[py_unflipped, px] if 0 <= py_unflipped < height and 0 <= px < width else -1
    val_flipped = img[py_flipped, px] if 0 <= py_flipped < height and 0 <= px < width else -1
    
    return px, py_unflipped, val_unflipped, py_flipped, val_flipped

points = [
    [7.595, -3.576], # Where it crashed
    [7.5, -3.5],     # Spawn point
    [-2.2, -3.5],    # Another spawn point
]

for p in points:
    px, py1, v1, py2, v2 = get_pixel(p[0], p[1])
    print(f"World ({p[0]}, {p[1]}):")
    print(f"  Unflipped Pixel ({px}, {py1}) -> Value={v1}")
    print(f"  Flipped Pixel ({px}, {py2}) -> Value={v2}")
