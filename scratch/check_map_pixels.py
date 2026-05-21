
import yaml
import numpy as np
from PIL import Image
import os

def check_map_values():
    map_yaml = "/home/yuta775/projects/f1tenth-rl-project/my_maps/testmap-tamoku/map-tamoku.yaml"
    with open(map_yaml, 'r') as f:
        map_conf = yaml.safe_load(f)

    origin = map_conf['origin']
    resolution = map_conf['resolution']
    img_name = map_conf['image']
    img_path = os.path.join(os.path.dirname(map_yaml), img_name)

    img = Image.open(img_path)
    map_img = np.array(img)
    height, width = map_img.shape

    poses = [
        [-3, -3.5],
        [7, -3.7],
    ]

    for x, y in poses:
        px = int((x - origin[0]) / resolution)
        py = int(height - (y - origin[1]) / resolution)
        print(f"\nPose ({x}, {y}) -> Pixel ({px}, {py})")
        
        # Check 10x10 area
        patch = map_img[py-5:py+5, px-5:px+5]
        print(patch)

if __name__ == "__main__":
    check_map_values()
