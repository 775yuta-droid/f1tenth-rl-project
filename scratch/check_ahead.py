import sys
import os
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src import config
from src.f1_env import F1TenthRL

env = F1TenthRL(config.MAP_PATH)
sim = env.env.sim

# [-2.2, 3.5]
x = -2.2
y = 3.5

# pixels
c = int((x - sim.map_origin[0]) / sim.map_resolution)
r = int((y - sim.map_origin[1]) / sim.map_resolution)

print(f"Spawn pixel: r={r}, c={c}")

# check pixels ahead (+X direction)
for i in range(20):
    # check row r, col c+i
    if r < sim.map_height and c+i < sim.map_width:
        val = sim.map_img[r, c+i]
        print(f"X={x + i*0.05:.2f} (col {c+i}): val={val}")
