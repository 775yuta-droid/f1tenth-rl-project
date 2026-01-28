import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from PIL import Image

# 共通モジュールから config を読み込む
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def view_spawn():
    map_yaml_path = config.MAP_PATH + ".yaml"
    
    # パスの補正 (Docker内パス /workspace/.. をローカルパスに変換、またはその逆の考慮)
    # ここではローカル実行を想定して相対パス調整を試みる
    if not os.path.exists(map_yaml_path):
        # config.MAP_PATH が /workspace/my_maps/my_map の場合
        # プロジェクトルートから実行している場合、 ./my_maps/my_map.yaml になる
        local_map_path = map_yaml_path.replace('/workspace/', './')
        if os.path.exists(local_map_path):
            map_yaml_path = local_map_path
        else:
            print(f"Error: Map file not found at {map_yaml_path}. \nChecked local path: {local_map_path}")
            return

    # Load YAML
    with open(map_yaml_path, 'r') as f:
        map_config = yaml.safe_load(f)
    
    img_name = map_config['image']
    resolution = map_config['resolution']
    origin = map_config['origin'] # [x, y, theta]
    
    # Load Image
    map_dir = os.path.dirname(map_yaml_path)
    img_path = os.path.join(map_dir, img_name)
    img = Image.open(img_path)
    img_arr = np.array(img)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_arr, cmap='gray', origin='lower') # origin='lower' is important for map/image coordination if flipped
    # Note: ROS map coordinate system vs Image coordinate system can be tricky.
    # Usually pgm is loaded top-left (0,0) in python image lib, but ROS uses bottom-left.
    # let's just plot naturally and adjust standard coordinate transform.
    
    # World coords to Pixel coords
    # pixel_x = (world_x - origin_x) / resolution
    # pixel_y = (world_y - origin_y) / resolution  (if map is not flipped)
    # Note: If origin='lower' in imshow, then (0,0) pixel is bottom-left.
    # But Image.open loads (0,0) as top-left.
    # So we should use origin='upper' (default) and flip Y calc, OR flip array.
    # Let's use standard image coords (top-left is 0,0) for the background.
    ax.imshow(img_arr, cmap='gray') 
    
    # Height is needed to flip Y for display if we treat (0,0) as bottom-left world origin
    height, width = img_arr.shape
    
    # Parse Start Pose
    start_x, start_y, start_yaw = config.START_POSE
    
    # Transform World -> Image
    # image_x = (world_x - origin_x) / resolution
    # image_y = height - (world_y - origin_y) / resolution
    
    px = (start_x - origin[0]) / resolution
    py = height - (start_y - origin[1]) / resolution
    
    print(f"Start Pose: {config.START_POSE}")
    print(f"Pixel Coords: ({px:.1f}, {py:.1f})")
    
    # Check bounds
    if 0 <= px < width and 0 <= py < height:
        pixel_value = img_arr[int(py), int(px)]
        # Assuming PGM: 255=white(free), 0=black(occupied), 205=unknown
        status = "UNKNOWN/OBSTACLE"
        if pixel_value > 250: status = "FREE SPACE (SAFE)"
        elif pixel_value < 10: status = "OBSTACLE (COLLISION)"
        
        print(f"Location Status: {status} (Pixel Value: {pixel_value})")
        
        # Draw Point
        ax.plot(px, py, 'ro', markersize=10, label='Start')
        
        # Draw Direction Arrow
        arrow_len = 20 # pixels
        dx = arrow_len * np.cos(start_yaw)
        dy = arrow_len * np.sin(start_yaw) 
        # Note: raw image y-axis increases downwards. 
        # Positive Y in world is UP. So in image pixels, positive Y change is NEGATIVE pixel Y change.
        # But we already flipped Y in py calculation.
        # Let's adjust arrow dy:
        # If yaw is 90 deg (pi/2), car points UP. In image, that is negative Y.
        ax.arrow(px, py, dx, -dy, head_width=10, head_length=10, fc='r', ec='r')
        
    else:
        print("WARNING: Coordinate is OUT of map bounds!")
    
    ax.set_title(f"Start Pose Preview: {config.START_POSE}\nres={resolution}, origin={origin}")
    ax.legend()
    
    # Save
    save_path = "spawn_preview.png"
    plt.savefig(save_path)
    print(f"Saved preview to: {save_path}")
    plt.close()

if __name__ == "__main__":
    view_spawn()
