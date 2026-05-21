import numpy as np
from PIL import Image
import yaml
import os

MAP_PATH = "/workspace/my_maps/testmap-tamoku/map-tamoku"
map_yaml = MAP_PATH + ".yaml"
map_pgm = MAP_PATH + ".pgm"

with open(map_yaml, 'r') as f:
    meta = yaml.safe_load(f)

res = meta['resolution']
origin = meta['origin']

img = np.array(Image.open(map_pgm).transpose(Image.FLIP_TOP_BOTTOM))

def get_val(x, y):
    c = int((x - origin[0]) / res)
    r = int((y - origin[1]) / res)
    if 0 <= r < img.shape[0] and 0 <= c < img.shape[1]:
        return img[r, c]
    return -1

# [-3.0, 1.22] 周辺の 5x5 ピクセルを表示
print(f"Origin: {origin}, Res: {res}")
print("Map values around [-3.0, 1.22]:")
for dy in np.arange(0.1, -0.15, -0.05):
    line = []
    for dx in np.arange(-0.1, 0.15, 0.05):
        val = get_val(-3.0 + dx, 1.22 + dy)
        line.append(f"{val:3}")
    print(" ".join(line))
