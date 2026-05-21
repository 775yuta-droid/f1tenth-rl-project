import sys
import os
import numpy as np
from PIL import Image
import yaml
import scipy.ndimage as ndimage

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from src import config

def find_moderate_safe_spawns():
    map_yaml = config.MAP_PATH + ".yaml"
    map_pgm = config.MAP_PATH + ".pgm"
    
    with open(map_yaml, 'r') as f:
        meta = yaml.safe_load(f)
    
    res = meta['resolution']
    origin = meta['origin']
    
    img = np.array(Image.open(map_pgm).transpose(Image.FLIP_TOP_BOTTOM))
    height, width = img.shape
    
    # 距離変換 (壁からの距離)
    # 0: 壁, 255: 走行可能
    dt = ndimage.distance_transform_edt(img == 255)
    
    # 壁から 0.5m 〜 1.5m の範囲を探す (コース幅が 2-3m と想定)
    # これにより、コース外の「壁から遠すぎる領域」を除外できる可能性がある
    target_mask = (dt > (0.5 / res)) & (dt < (1.5 / res))
    
    # さらに、ユーザー様の座標 [-2.2, 3.5] に比較的近いコンポーネントを選ぶ
    # (これにより、コース内の白領域に限定する)
    labeled, num_features = ndimage.label(img == 255)
    start_c = int((-2.2 - origin[0]) / res)
    start_r = int((3.5 - origin[1]) / res)
    if start_r < height and start_c < width:
        track_label = labeled[start_r, start_c]
        target_mask = target_mask & (labeled == track_label)
    
    safe_points = np.argwhere(target_mask)
    print(f"候補ピクセル数: {len(safe_points)}")
    
    if len(safe_points) == 0:
        print("条件に合う地点が見つかりません。")
        return
        
    print("安全なワールド座標候補 (yaw=0.0):")
    np.random.seed(42)
    indices = np.random.choice(len(safe_points), min(5, len(safe_points)), replace=False)
    for idx in indices:
        r, c = safe_points[idx]
        wx = origin[0] + c * res
        wy = origin[1] + r * res
        dist_m = dt[r, c] * res
        print(f"[{wx:.2f}, {wy:.2f}, 0.0]  (壁までの距離: {dist_m:.2f} m)")

if __name__ == "__main__":
    find_moderate_safe_spawns()
