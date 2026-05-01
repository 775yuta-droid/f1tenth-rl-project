
import yaml
import numpy as np
from PIL import Image
import os

def check_spawns():
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

    # free_thresh is usually 0.196 in ROS (values < 255*0.196 are occupied? No, usually 0 is occupied, 255 is free in grayscale)
    # The YAML says occupied_thresh: 0.65, free_thresh: 0.25
    # If negate is 0:
    #   p = (255 - x) / 255.0
    #   p > occupied_thresh => occupied
    #   p < free_thresh => free
    # In PGM, white (255) is free, black (0) is occupied.
    # So x=255 => p=0 < 0.25 => free.
    # x=0 => p=1 > 0.65 => occupied.

    start_poses = [
        [-3, -3.5, 0.0],
        [7, -3.7, 0.3],
    ]

    noise_range = 0.1
    car_length = 0.465
    car_width = 0.19

    print(f"Map Size: {width}x{height}, Resolution: {resolution}")
    
    for i, pose in enumerate(start_poses):
        x_base, y_base, yaw = pose
        print(f"\nSpawn #{i}: ({x_base}, {y_base}) yaw={yaw}")
        
        # Check a grid around the spawn point including noise and car size
        # We'll check the bounding box of the car at the base pose, and then expand it by noise.
        # For simplicity, let's check a box of size (car_length + 2*noise) x (car_width + 2*noise)
        check_r = max(car_length, car_width) / 2.0 + noise_range
        
        steps = 10
        for dx in np.linspace(-check_r, check_r, steps):
            for dy in np.linspace(-check_r, check_r, steps):
                x = x_base + dx
                y = y_base + dy
                
                px = int((x - origin[0]) / resolution)
                py = int(height - (y - origin[1]) / resolution)
                
                if 0 <= px < width and 0 <= py < height:
                    val = map_img[py, px]
                    p = (255 - val) / 255.0
                    if p > 0.25: # Not free
                        print(f"  [!] Potential Collision at rel ({dx:.2f}, {dy:.2f}) -> px({px, py}) val={val} p={p:.2f}")
                else:
                    print(f"  [!] Out of bounds at rel ({dx:.2f}, {dy:.2f})")

if __name__ == "__main__":
    check_spawns()
