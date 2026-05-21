
import yaml
import numpy as np
from PIL import Image
import os

def measure_track_width():
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

    for i, (x, y) in enumerate(poses):
        px_center = int((x - origin[0]) / resolution)
        py_center = int(height - (y - origin[1]) / resolution)
        
        print(f"\nSpawn #{i}: ({x}, {y}) -> Pixel ({px_center}, {py_center})")
        
        # Search for walls in 4 directions
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        min_dist = 999
        for dx, dy in directions:
            dist = 0
            while True:
                dist += 1
                px = px_center + dx * dist
                py = py_center + dy * dist
                if not (0 <= px < width and 0 <= py < height):
                    break
                val = map_img[py, px]
                p = (255 - val) / 255.0
                if p > 0.25: # Wall
                    break
            
            real_dist = dist * resolution
            print(f"  Wall at direction {dx, dy}: {real_dist:.3f} m ({dist} pixels)")
            if real_dist < min_dist:
                min_dist = real_dist
        
        print(f"  Closest wall: {min_dist:.3f} m")
        car_half_width = 0.19 / 2.0
        margin = min_dist - car_half_width
        print(f"  Safety Margin (from half-width 0.095m): {margin:.3f} m")
        print(f"  Current Noise Range: ±0.1 m")
        if margin < 0.1:
            print(f"  [!!!] DANGER: Noise can push car into wall!")

if __name__ == "__main__":
    measure_track_width()
