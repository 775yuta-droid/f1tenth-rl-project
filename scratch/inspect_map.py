import yaml
import numpy as np
from PIL import Image
import os

map_yaml = "my_maps/testmap-tamoku/map-tamoku.yaml"
map_pgm = "my_maps/testmap-tamoku/map-tamoku.pgm"

with open(map_yaml, 'r') as f:
    config = yaml.safe_load(f)

img = np.array(Image.open(map_pgm))
# 0 is black, 255 is white.
# free_thresh = 0.25 (i.e. pixel > 255*(1-0.25) = 191 is free?)
# Wait, standard ROS maps:
# p > occ_thresh -> occupied (black)
# p < free_thresh -> free (white)
# The image values are typically 0 to 255.
# Let's just count unique values.
unique, counts = np.unique(img, return_counts=True)
print("Unique pixel values and counts:")
for u, c in zip(unique, counts):
    print(f"  {u}: {c}")

