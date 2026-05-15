"""
改良版：マップ中心線（レーシングライン）自動生成スクリプト

ループコースに対応し、ノイズ除去と閾値の厳密な適用（白部分のみを抽出）を行います。
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
    # YAMLの閾値を読み込む
    occ_thresh = float(meta.get("occupied_thresh", 0.65))
    free_thresh = float(meta.get("free_thresh", 0.25))

    if negate == 0:
        prob = 1.0 - img / 255.0
    else:
        prob = img / 255.0

    return prob, resolution, origin, free_thresh


def prune_skeleton(skeleton: np.ndarray):
    """スケルトンの短い枝（ノイズ）を除去"""
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
            for r in range(2, 5): # 探索範囲を少し広げる
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
    parser.add_argument("--free_thresh", type=float, default=None, 
                        help="走行可能とみなす閾値（デフォルトはYAML値。島を突き抜ける場合は0.1等に下げる）")
    args = parser.parse_args()

    prob, res, origin, yaml_free_thresh = load_map(args.map)
    H, W = prob.shape

    # 閾値の決定（指定がなければYAML値を使うが、205(prob=0.19)を避けるため厳しめに設定可能にする）
    free_thresh = args.free_thresh if args.free_thresh is not None else yaml_free_thresh
    print(f"[INFO] Free Thresh: {free_thresh} (YAML値: {yaml_free_thresh})")

    print("[1/4] コース領域を抽出中（白部分のみ）...")
    # 確実に走行可能なエリア（白）のみを抽出
    free_mask = (prob < free_thresh)
    
    # マップに205(グレー)が含まれている場合、free_thresh=0.25だと突き抜けるため
    # 自動的に厳しめの判定を行う（もし254と205が混在しているなら、その中間を狙う）
    if free_thresh == 0.25 and np.any((prob > 0.15) & (prob < 0.22)):
        print("[WARNING] マップにグレー領域を検出しました。閾値を 0.1 に下げてコースを限定します。")
        free_mask = (prob < 0.1)

    skeleton = skeletonize(free_mask)
    skeleton = prune_skeleton(skeleton)

    # スタート位置の推定（デフォルト 0.04, 3.68）
    start_row = int((H - 1) - (3.68 - origin[1]) / res)
    start_col = int((0.04 - origin[0]) / res)

    print("[2/4] 点列を順序付け中...")
    ordered = order_points_robust(skeleton, (start_row, start_col))
    
    if len(ordered) < 10:
        print("エラー: 抽出されたパスが短すぎます。閾値設定 (--free_thresh) を確認してください。")
        return

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
        print("[INFO] ループを接続します。")
        xy_world[-1] = xy_world[0]

    # ヘディングと曲率
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
