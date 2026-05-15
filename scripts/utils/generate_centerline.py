"""
改良版：マップ中心線（レーシングライン）自動生成スクリプト

ループコースに対応し、ノイズ（枝分かれ）を除去して綺麗なセンターラインを抽出します。
"""

import argparse
import os
import sys
import numpy as np
import csv
from typing import List, Tuple, Set

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config

try:
    from PIL import Image
    from scipy import ndimage
    from skimage.morphology import skeletonize, remove_small_objects
except ImportError as e:
    print(f"[ERROR] 必要なライブラリが不足しています: {e}")
    sys.exit(1)


def load_map(map_path: str):
    import yaml
    pgm_path  = map_path + ".pgm"
    yaml_path = map_path + ".yaml"

    img = np.array(Image.open(pgm_path).convert("L"))
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    resolution = float(meta["resolution"])
    origin     = meta["origin"][:2]
    negate     = int(meta.get("negate", 0))
    occ_thresh = float(meta.get("occupied_thresh", 0.65))

    if negate == 0:
        prob = 1.0 - img / 255.0
    else:
        prob = img / 255.0

    binary = (prob > occ_thresh).astype(np.uint8)
    return binary, resolution, origin


def prune_skeleton(skeleton: np.ndarray, min_dist: int = 10):
    """
    スケルトンの「ひげ（行き止まりの短い枝）」を除去する。
    行き止まり（次数1）から遡って、分岐点に当たるまで、または一定距離まで削る。
    """
    skel = skeleton.copy().astype(np.uint8)
    
    while True:
        # 各点の隣接数を計算
        kernel = np.array([[1,1,1], [1,0,1], [1,1,1]])
        neighbors_count = ndimage.convolve(skel, kernel, mode='constant') * skel
        
        # 行き止まり（隣接1）の点を抽出
        endpoints = np.argwhere(neighbors_count == 1)
        if len(endpoints) == 0:
            break
            
        changed = False
        for r, c in endpoints:
            # 周囲3x3を見て、接続先が分岐点（隣接>2）でなければ削る
            # ここでは単純化のため、行き止まりを1ピクセルずつ削る処理を繰り返す
            skel[r, c] = 0
            changed = True
            
        if not changed:
            break
            
    return skel.astype(bool)


def order_points_robust(skeleton: np.ndarray, start_hint_px: Tuple[int, int]):
    """
    スタート地点に近い点から開始し、最も近い未訪問の点を探して順序付ける。
    """
    points = np.argwhere(skeleton)
    if len(points) == 0:
        return []

    point_set = set(map(tuple, points))
    
    # スタート地点に最も近いスケルトン上の点を探す
    start_px = min(point_set, key=lambda p: (p[0]-start_hint_px[0])**2 + (p[1]-start_hint_px[1])**2)
    
    ordered = []
    curr = start_px
    
    while curr and point_set:
        ordered.append(curr)
        point_set.remove(curr)
        
        # 隣接する未訪問点を探す (3x3)
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nb = (curr[0] + dr, curr[1] + dc)
                if nb in point_set:
                    neighbors.append(nb)
        
        if neighbors:
            # 複数ある場合は、とりあえず最初の1つ（ノイズがなければ1つのはず）
            curr = neighbors[0]
        else:
            # 少し離れた場所を探す (5x5)
            found = False
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nb = (curr[0] + dr, curr[1] + dc)
                    if nb in point_set:
                        curr = nb
                        found = True
                        break
                if found: break
            if not found:
                curr = None
                
    return ordered


def compute_heading_curvature(xy_world, is_loop=True):
    # ループの場合、端の影響を抑えるためにデータを拡張する
    if is_loop and len(xy_world) > 10:
        data = np.vstack([xy_world[-5:], xy_world, xy_world[:5]])
    else:
        data = xy_world

    dx = np.gradient(data[:, 0])
    dy = np.gradient(data[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    heading = np.arctan2(dy, dx)
    denom = (dx**2 + dy**2) ** 1.5 + 1e-9
    curvature = (dx * ddy - dy * ddx) / denom

    if is_loop and len(xy_world) > 10:
        return heading[5:-5], curvature[5:-5]
    return heading, curvature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default=config.MAP_PATH)
    parser.add_argument("--start_x", type=float, default=0.04) # ユーザー指定のスタート位置
    parser.add_argument("--start_y", type=float, default=3.68)
    args = parser.parse_args()

    binary, res, origin = load_map(args.map)
    H, W = binary.shape

    print("[1/4] 骨格線を抽出中...")
    skeleton = skeletonize(binary == 0)
    
    print("[2/4] ノイズ（枝分かれ）を除去中...")
    skeleton = prune_skeleton(skeleton)

    # スタート地点の座標変換
    start_row = int((H - 1) - (args.start_y - origin[1]) / res)
    start_col = int((args.start_x - origin[0]) / res)

    print("[3/4] 点列を順序付け中...")
    ordered = order_points_robust(skeleton, (start_row, start_col))
    
    if not ordered:
        print("エラー: 点が見つかりませんでした")
        return

    # 世界座標変換
    xy_pixel = np.array(ordered)
    xy_world = np.zeros_like(xy_pixel, dtype=float)
    xy_world[:, 0] = origin[0] + xy_pixel[:, 1] * res
    xy_world[:, 1] = origin[1] + (H - 1 - xy_pixel[:, 0]) * res

    # 平滑化
    from scipy.signal import savgol_filter
    window = min(21, len(xy_world) // 2 * 2 + 1)
    if window > 3:
        xy_world[:, 0] = savgol_filter(xy_world[:, 0], window, 3)
        xy_world[:, 1] = savgol_filter(xy_world[:, 1], window, 3)

    # ループの終端を始端に接続
    is_loop = np.linalg.norm(xy_world[0] - xy_world[-1]) < 2.0
    if is_loop:
        print("[INFO] ループを検出。終端を接続します。")
        xy_world[-1] = xy_world[0]

    print("[4/4] ヘディングと曲率を計算中...")
    heading, curvature = compute_heading_curvature(xy_world, is_loop)

    output_path = args.map + "_centerline.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "heading", "curvature"])
        for i in range(len(xy_world)):
            writer.writerow([f"{xy_world[i,0]:.4f}", f"{xy_world[i,1]:.4f}", f"{heading[i]:.6f}", f"{curvature[i]:.6f}"])

    print(f"完了: {output_path} ({len(xy_world)} points)")

if __name__ == "__main__":
    main()
