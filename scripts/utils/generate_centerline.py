"""
改良版：マップ中心線（レーシングライン）自動生成スクリプト

ループコース・逆走指定対応版
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
    from skimage.morphology import skeletonize
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
    free_thresh = float(meta.get("free_thresh", 0.25))

    if negate == 0:
        prob = 1.0 - img / 255.0
    else:
        prob = img / 255.0

    return prob, resolution, origin, free_thresh


def prune_skeleton(skeleton: np.ndarray):
    skel = skeleton.copy().astype(np.uint8)
    while True:
        kernel = np.array([[1,1,1], [1,0,1], [1,1,1]])
        neighbors_count = ndimage.convolve(skel, kernel, mode='constant') * skel
        endpoints = np.argwhere(neighbors_count == 1)
        if len(endpoints) == 0: break
        changed = False
        for r, c in endpoints:
            skel[r, c] = 0
            changed = True
        if not changed: break
    return skel.astype(bool)


def order_points_robust(skeleton: np.ndarray, start_hint_px: Tuple[int, int]):
    points = np.argwhere(skeleton)
    if len(points) == 0: return []
    point_set = set(map(tuple, points))
    start_px = min(point_set, key=lambda p: (p[0]-start_hint_px[0])**2 + (p[1]-start_hint_px[1])**2)
    ordered = []
    curr = start_px
    while curr and point_set:
        ordered.append(curr)
        point_set.remove(curr)
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nb = (curr[0] + dr, curr[1] + dc)
                if nb in point_set: neighbors.append(nb)
        if neighbors:
            curr = neighbors[0]
        else:
            found = False
            for r in range(2, 6):
                for dr in range(-r, r+1):
                    for dc in range(-r, r+1):
                        nb = (curr[0] + dr, curr[1] + dc)
                        if nb in point_set:
                            curr = nb; found = True; break
                    if found: break
                if found: break
            if not found: curr = None
    return ordered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default=config.MAP_PATH)
    parser.add_argument("--reverse", action="store_true", help="進行方向を逆にする")
    parser.add_argument("--mask-box", type=float, nargs=4, action="append", metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                        help="除外したい矩形領域の物理座標 [xmin, xmax, ymin, ymax] (複数指定可)")
    parser.add_argument("--min-width", type=float, default=0.0,
                        help="最小道路幅（メートル）。これより狭い道を塞ぎます。")
    args = parser.parse_args()

    prob, res, origin, yaml_free_thresh = load_map(args.map)
    H, W = prob.shape

    # グレー領域対策
    free_mask = (prob < 0.1) if np.any((prob > 0.15) & (prob < 0.22)) else (prob < yaml_free_thresh)

    # 道路幅によるフィルタリング
    if args.min_width > 0.0:
        print(f"[INFO] 道路幅が {args.min_width}m 未満の経路を除外します。")
        dt = ndimage.distance_transform_edt(free_mask)
        # 距離変換の値は半径に相当するため、直径（道路幅）= dt * 2 * res
        min_width_px = args.min_width / (2.0 * res)
        free_mask = free_mask & (dt >= min_width_px)

    # 特定の物理座標領域をマスクする
    if args.mask_box:
        for box in args.mask_box:
            xmin, xmax, ymin, ymax = box
            print(f"[INFO] 物理座標 x:[{xmin}, {xmax}], y:[{ymin}, {ymax}] の領域を除外します。")
            
            # 物理座標から画像ピクセルインデックスへの変換
            r_min = int((H - 1) - (ymax - origin[1]) / res)
            r_max = int((H - 1) - (ymin - origin[1]) / res)
            c_min = int((xmin - origin[0]) / res)
            c_max = int((xmax - origin[0]) / res)
            
            # インデックスの順序を補正して画像サイズ内にクリップ
            r_start = max(0, min(r_min, r_max))
            r_end = min(H, max(r_min, r_max))
            c_start = max(0, min(c_min, c_max))
            c_end = min(W, max(c_min, c_max))
            
            # 指定された範囲を障害物(False)にする
            free_mask[r_start:r_end, c_start:c_end] = False

    print("[1/3] 骨格線を抽出中...")
    skeleton = prune_skeleton(skeletonize(free_mask))

    # スタート位置 (デフォルト 0.04, 3.68)
    start_row = int((H - 1) - (3.68 - origin[1]) / res)
    start_col = int((0.04 - origin[0]) / res)

    print("[2/3] 点列を順序付け中...")
    ordered = order_points_robust(skeleton, (start_row, start_col))
    
    if not ordered:
        print("[ERROR] センターラインの点が検出されませんでした。")
        print("以下の原因が考えられます：")
        print("1. マスク範囲 (--mask-box) が広すぎて、スタート位置やコース全体を塞いでしまっている。")
        print("2. 最小道路幅 (--min-width) が大きすぎて、コースの狭い部分でループが分断されている。")
        print("引数や座標の指定内容を確認してください。")
        sys.exit(1)
    
    if args.reverse:
        print("[INFO] 進行方向を逆にします。")
        ordered = ordered[::-1]

    xy_pixel = np.array(ordered)
    xy_world = np.zeros_like(xy_pixel, dtype=float)
    xy_world[:, 0] = origin[0] + xy_pixel[:, 1] * res
    xy_world[:, 1] = origin[1] + (H - 1 - xy_pixel[:, 0]) * res

    from scipy.signal import savgol_filter
    window = min(15, len(xy_world) // 2 * 2 + 1)
    if window > 3:
        xy_world[:, 0] = savgol_filter(xy_world[:, 0], window, 3)
        xy_world[:, 1] = savgol_filter(xy_world[:, 1], window, 3)

    is_loop = np.linalg.norm(xy_world[0] - xy_world[-1]) < 2.0
    if is_loop:
        xy_world[-1] = xy_world[0]

    print("[3/3] 物理量を計算中...")
    data = np.vstack([xy_world[-5:], xy_world, xy_world[:5]]) if is_loop else xy_world
    dx = np.gradient(data[:, 0]); dy = np.gradient(data[:, 1])
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    heading = np.arctan2(dy, dx)
    denom = (dx**2 + dy**2) ** 1.5 + 1e-9
    curvature = (dx * ddy - dy * ddx) / denom
    if is_loop:
        heading = heading[5:-5]; curvature = curvature[5:-5]

    output_path = args.map + "_centerline.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "heading", "curvature"])
        for i in range(len(xy_world)):
            writer.writerow([f"{xy_world[i,0]:.4f}", f"{xy_world[i,1]:.4f}", f"{heading[i]:.6f}", f"{curvature[i]:.6f}"])

    print(f"完了: {output_path} ({len(xy_world)} points)")

if __name__ == "__main__":
    main()
