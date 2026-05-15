"""
マップ中心線（レーシングライン）自動生成スクリプト

PGMマップファイルから走路の中心線ウェイポイントを抽出し、
CSVとして保存します。

生成されるファイル: <map_name>_centerline.csv
列: x[m], y[m], heading[rad], curvature[1/m]

使い方:
    python3 scripts/utils/generate_centerline.py
    python3 scripts/utils/generate_centerline.py --map /workspace/my_maps/honbann-cose/map_1_0509_145516
"""

import argparse
import os
import sys
import numpy as np
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config

try:
    from PIL import Image
    from scipy import ndimage
    from skimage.morphology import skeletonize
    from skimage.graph import route_through_array
except ImportError as e:
    print(f"[ERROR] 必要なライブラリが不足しています: {e}")
    print("  pip install pillow scipy scikit-image")
    sys.exit(1)


def load_map(map_path: str):
    """PGMとYAMLからマップ画像とメタデータを読み込む"""
    import yaml
    pgm_path  = map_path + ".pgm"
    yaml_path = map_path + ".yaml"

    img = np.array(Image.open(pgm_path).convert("L"))

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    resolution = float(meta["resolution"])      # m/pixel
    origin     = meta["origin"][:2]             # [x_m, y_m]
    negate     = int(meta.get("negate", 0))
    occ_thresh = float(meta.get("occupied_thresh", 0.65))
    free_thresh = float(meta.get("free_thresh", 0.25))

    # pgm 輝度 → 占有確率 (0=free, 1=occupied)
    # negate=0 のとき: 輝度高=free (255=white=free)
    if negate == 0:
        prob = 1.0 - img / 255.0
    else:
        prob = img / 255.0

    # 二値化: occupied=1, free=0
    binary = (prob > occ_thresh).astype(np.uint8)

    return binary, resolution, origin


def pixel_to_world(px, py, resolution, origin):
    """ピクセル座標(px,py) → ワールド座標(x_m, y_m)"""
    x_m = origin[0] + px * resolution
    # PGMは上下反転（行0=地図上端）
    y_m = origin[1] + py * resolution
    return x_m, y_m


def extract_skeleton_centerline(binary_map):
    """
    走路領域（free space）の骨格線を求める

    Returns:
        skeleton: bool配列 (H, W)
    """
    # free space: binary==0 の領域
    free = (binary_map == 0).astype(np.uint8)

    # モルフォロジー骨格化
    skeleton = skeletonize(free > 0)
    return skeleton


def order_skeleton_points(skeleton, start_hint=None):
    """
    骨格線の連続した点列を取り出す（幅優先で最長経路を抽出）

    Returns:
        ordered: [(row, col), ...] のリスト
    """
    points = np.argwhere(skeleton)  # (N, 2) = [(row, col), ...]

    if len(points) == 0:
        raise ValueError("骨格線の点が見つかりませんでした")

    # 隣接グラフを構築してDFSで最長連続経路を抽出
    point_set = set(map(tuple, points))

    def neighbors(r, c):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in point_set:
                    yield (r + dr, c + dc)

    # 端点を探す（隣接点が1個の点）
    endpoints = []
    for r, c in point_set:
        n = sum(1 for _ in neighbors(r, c))
        if n == 1:
            endpoints.append((r, c))

    start = endpoints[0] if endpoints else points[0].tolist()
    start = tuple(start)

    # 貪欲DFSで順序付け
    visited = set()
    ordered = []
    stack = [start]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        ordered.append(cur)
        for nb in neighbors(*cur):
            if nb not in visited:
                stack.append(nb)

    return ordered


def smooth_path(points, window=15):
    """移動平均で平滑化"""
    pts = np.array(points, dtype=float)
    result = np.zeros_like(pts)
    half = window // 2
    for i in range(len(pts)):
        lo = max(0, i - half)
        hi = min(len(pts), i + half + 1)
        result[i] = pts[lo:hi].mean(axis=0)
    return result


def compute_heading_curvature(xy_world):
    """
    x,y 列からヘディングと曲率を計算する

    Args:
        xy_world: (N, 2) numpy array [x_m, y_m]

    Returns:
        heading:   (N,) array [rad]
        curvature: (N,) array [1/m]
    """
    dx = np.gradient(xy_world[:, 0])
    dy = np.gradient(xy_world[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    heading = np.arctan2(dy, dx)

    # κ = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
    denom = (dx**2 + dy**2) ** 1.5 + 1e-9
    curvature = (dx * ddy - dy * ddx) / denom

    return heading, curvature


def save_centerline(output_path, xy_world, heading, curvature):
    """CSVに保存"""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "heading", "curvature"])
        for i in range(len(xy_world)):
            writer.writerow([
                f"{xy_world[i, 0]:.4f}",
                f"{xy_world[i, 1]:.4f}",
                f"{heading[i]:.6f}",
                f"{curvature[i]:.6f}",
            ])
    print(f"[OK] 保存: {output_path} ({len(xy_world)} 点)")


def main():
    parser = argparse.ArgumentParser(description="マップ中心線の自動生成")
    parser.add_argument("--map", type=str, default=config.MAP_PATH,
                        help="マップファイルのパス（拡張子なし）")
    parser.add_argument("--smooth", type=int, default=20,
                        help="平滑化ウィンドウサイズ (デフォルト: 20)")
    parser.add_argument("--downsample", type=int, default=5,
                        help="ウェイポイントの間引き (デフォルト: 5点に1点)")
    args = parser.parse_args()

    map_path = args.map
    print(f"[INFO] マップ: {map_path}")

    binary, resolution, origin = load_map(map_path)
    print(f"[INFO] マップサイズ: {binary.shape}, 解像度: {resolution} m/px, 原点: {origin}")

    print("[INFO] 骨格線を抽出中...")
    skeleton = extract_skeleton_centerline(binary)
    print(f"[INFO] 骨格線点数: {skeleton.sum()}")

    print("[INFO] 点列を順序付け中...")
    ordered = order_skeleton_points(skeleton)
    print(f"[INFO] 順序付き点数: {len(ordered)}")

    # 座標変換: (row, col) → world (x, y)
    # PGMはrow=0が上端なので、y方向に反転する
    # world_y = origin[1] + (H - 1 - row) * resolution
    H = binary.shape[0]
    xy_pixel = np.array(ordered, dtype=float)  # (N, 2) = [row, col]
    xy_world = np.zeros((len(ordered), 2), dtype=float)
    xy_world[:, 0] = origin[0] + xy_pixel[:, 1] * resolution       # x = origin_x + col * res
    xy_world[:, 1] = origin[1] + (H - 1 - xy_pixel[:, 0]) * resolution  # y (上下反転)

    print(f"[INFO] 平滑化中 (window={args.smooth})...")
    xy_smooth = smooth_path(xy_world, window=args.smooth)

    # 間引き
    xy_ds = xy_smooth[::args.downsample]
    print(f"[INFO] 間引き後: {len(xy_ds)} ウェイポイント")

    heading, curvature = compute_heading_curvature(xy_ds)

    output_path = map_path + "_centerline.csv"
    save_centerline(output_path, xy_ds, heading, curvature)

    # 統計表示
    print(f"\n[統計]")
    print(f"  曲率: min={curvature.min():.4f}, max={curvature.max():.4f}, mean={np.abs(curvature).mean():.4f} [1/m]")
    print(f"  X範囲: {xy_ds[:,0].min():.2f} ~ {xy_ds[:,0].max():.2f} m")
    print(f"  Y範囲: {xy_ds[:,1].min():.2f} ~ {xy_ds[:,1].max():.2f} m")


if __name__ == "__main__":
    main()
