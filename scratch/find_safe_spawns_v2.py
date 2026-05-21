import sys
import os
import numpy as np
from PIL import Image
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from src import config

def find_truly_safe_spawns():
    map_yaml = config.MAP_PATH + ".yaml"
    map_pgm = config.MAP_PATH + ".pgm"
    
    with open(map_yaml, 'r') as f:
        meta = yaml.safe_load(f)
    
    res = meta['resolution']
    origin = meta['origin']
    
    # F1Tenth Gym flips the image vertically!
    img = np.array(Image.open(map_pgm).transpose(Image.FLIP_TOP_BOTTOM))
    height, width = img.shape
    
    # We want points far from walls.
    # Let's find white pixels (255)
    import scipy.ndimage as ndimage
    dt = ndimage.distance_transform_edt(img == 255)
    
    # Find pixels where distance to wall is at least 1.0 meters (20 pixels)
    safe_mask = dt > (1.0 / res)
    
    safe_points = np.argwhere(safe_mask)
    if len(safe_points) == 0:
        print("1.0m以上の安全マージンを持つ地点が見つかりません。条件を緩和します。")
        safe_mask = dt > (0.6 / res) # 0.6m
        safe_points = np.argwhere(safe_mask)
    
    print(f"見つかった安全なピクセル数: {len(safe_points)}")
    
    # Print some coordinates
    print("安全なワールド座標候補 (yaw=0.0):")
    np.random.seed(42)
    indices = np.random.choice(len(safe_points), min(5, len(safe_points)), replace=False)
    for idx in indices:
        r, c = safe_points[idx]
        # F1Tenth's conversion: c is X, r is Y (because image is flipped)
        wx = origin[0] + c * res
        wy = origin[1] + r * res
        dist_m = dt[r, c] * res
        print(f"[{wx:.2f}, {wy:.2f}, 0.0]  (壁までの距離: {dist_m:.2f} m)")

if __name__ == "__main__":
    find_truly_safe_spawns()
