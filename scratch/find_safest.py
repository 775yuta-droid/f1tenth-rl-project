import numpy as np
from PIL import Image
import yaml
import scipy.ndimage as ndimage
import os

MAP_PATH = "/workspace/my_maps/testmap-tamoku/map-tamoku"
map_yaml = MAP_PATH + ".yaml"
map_pgm = MAP_PATH + ".pgm"

with open(map_yaml, 'r') as f:
    meta = yaml.safe_load(f)

res = meta['resolution']
origin = meta['origin']

img = np.array(Image.open(map_pgm).transpose(Image.FLIP_TOP_BOTTOM))

# 壁からの距離
dt = ndimage.distance_transform_edt(img == 255)
max_idx = np.unravel_index(np.argmax(dt), dt.shape)
max_dist = dt[max_idx] * res

wx = origin[0] + max_idx[1] * res
wy = origin[1] + max_idx[0] * res

print(f"Max distance point: [{wx:.2f}, {wy:.2f}]")
print(f"Distance to wall: {max_dist:.2f} m")

# 周辺のピクセルを確認
r, c = max_idx
print(f"Pixel value at max point: {img[r, c]}")
