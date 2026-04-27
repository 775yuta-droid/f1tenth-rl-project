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

# すべての黒ピクセル (壁) の座標をリストアップ
walls = np.argwhere(img < 128)
print(f"Total wall pixels: {len(walls)}")

def check_near_walls(x, y, radius_m=2.0):
    px = (x - origin[0]) / res
    py = (y - origin[1]) / res
    radius_px = radius_m / res
    
    print(f"Checking walls around [{x}, {y}] (px: [{px:.1f}, {py:.1f}])")
    
    found = 0
    for wr, wc in walls:
        dist_px = np.sqrt((wr - py)**2 + (wc - px)**2)
        if dist_px < radius_px:
            dist_m = dist_px * res
            if found < 10:
                print(f"  Wall at [{origin[0] + wc*res:.2f}, {origin[1] + wr*res:.2f}] (dist: {dist_m:.3f}m)")
            found += 1
    
    print(f"Total walls within {radius_m}m: {found}")

check_near_walls(-3.0, 1.22)
