import sys
import os
import numpy as np
from PIL import Image
import yaml
import scipy.ndimage as ndimage

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from src import config

def find_track_spawns():
    map_yaml = config.MAP_PATH + ".yaml"
    map_pgm = config.MAP_PATH + ".pgm"
    
    with open(map_yaml, 'r') as f:
        meta = yaml.safe_load(f)
    
    res = meta['resolution']
    origin = meta['origin']
    
    # 画像の読み込みと反転
    img = np.array(Image.open(map_pgm).transpose(Image.FLIP_TOP_BOTTOM))
    height, width = img.shape
    
    # 白い部分(走行可能領域)のマスク
    free_space = (img == 255)
    
    # 既知の安全なコース内座標（ユーザー様が教えてくれた [-2.2, 3.5]）
    known_x = -2.2
    known_y = 3.5
    
    start_c = int((known_x - origin[0]) / res)
    start_r = int((known_y - origin[1]) / res)
    
    # flood fill的な連結成分ラベリングを用いて、コース内の白ピクセルのみを抽出する
    labeled, num_features = ndimage.label(free_space)
    track_label = labeled[start_r, start_c]
    
    if track_label == 0:
        print("エラー: 基準座標が壁の中、またはコース外です。")
        return
        
    track_mask = (labeled == track_label)
    print(f"全白ピクセル数: {np.sum(free_space)}, コース内ピクセル数: {np.sum(track_mask)}")
    
    # 距離変換 (track_mask内での距離ではなく、壁からの距離)
    # img != 255 が障害物
    obstacles = (img != 255)
    dt = ndimage.distance_transform_edt(~obstacles)
    
    # コース内で、壁から1.0m以上離れている場所
    safe_mask = (track_mask) & (dt > (1.0 / res))
    
    safe_points = np.argwhere(safe_mask)
    if len(safe_points) == 0:
        print("1.0m以上の安全マージンを持つ地点が見つかりません。0.6mに緩和します。")
        safe_mask = (track_mask) & (dt > (0.6 / res))
        safe_points = np.argwhere(safe_mask)
        if len(safe_points) == 0:
            print("0.6mも無理でした。0.3mに緩和します。")
            safe_mask = (track_mask) & (dt > (0.3 / res))
            safe_points = np.argwhere(safe_mask)
    
    print(f"見つかった真のコース内安全ピクセル数: {len(safe_points)}")
    
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
    find_track_spawns()
