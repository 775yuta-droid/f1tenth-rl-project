"""
改良版：センターライン可視化スクリプト（マップ背景付き）

CSVファイルを読み込み、元のマップ画像（PGM）の上に重ねて表示します。
座標系の不整合を修正（画像の上下反転を考慮）。
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import yaml
from PIL import Image

def load_map_info(csv_path):
    """CSVパスから対応するマップのPGMとYAMLを読み込む"""
    base_path = csv_path.replace("_centerline.csv", "")
    pgm_path = base_path + ".pgm"
    yaml_path = base_path + ".yaml"

    if not os.path.exists(pgm_path) or not os.path.exists(yaml_path):
        print(f"[Warning] マップファイルが見つかりません: {pgm_path} または {yaml_path}")
        return None, None, None

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    
    res = float(meta["resolution"])
    origin = meta["origin"][:2]
    
    img = np.array(Image.open(pgm_path).convert("L"))
    return img, res, origin

def plot_centerline_with_map(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return

    xs, ys, hs, ks = [], [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            hs.append(float(row["heading"]))
            ks.append(float(row["curvature"]))
    
    x = np.array(xs)
    y = np.array(ys)
    heading = np.array(hs)
    curvature = np.array(ks)

    img, res, origin = load_map_info(csv_path)

    plt.figure(figsize=(12, 12))

    if img is not None:
        # PGM画像(行0=上)をワールド座標系(下=y_min)に合わせるため上下反転
        img_flipped = np.flipud(img)
        h, w = img_flipped.shape
        extent = [
            origin[0], 
            origin[0] + w * res, 
            origin[1], 
            origin[1] + h * res
        ]
        plt.imshow(img_flipped, cmap='gray', extent=extent, origin='lower', alpha=0.6)

    # センターライン
    sc = plt.scatter(x, y, c=np.abs(curvature), cmap='jet', s=15, label='Waypoints', zorder=3)
    plt.colorbar(sc, label='Curvature [1/m]', shrink=0.8)
    plt.plot(x, y, 'cyan', alpha=0.4, linewidth=1.5, zorder=2)

    # 開始・終了
    plt.plot(x[0], y[0], 'go', markersize=10, label='Start', zorder=4)
    plt.plot(x[-1], y[-1], 'ro', markersize=10, label='End', zorder=4)

    # 進行方向
    step = max(1, len(x) // 30)
    plt.quiver(x[::step], y[::step], 
               np.cos(heading[::step]), np.sin(heading[::step]),
               color='white', scale=30, width=0.003, alpha=0.8, zorder=5)

    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title(f'Centerline on Map: {os.path.basename(csv_path)}')
    plt.legend(loc='upper right')

    output_img = csv_path.replace('.csv', '_with_map.png')
    plt.savefig(output_img, bbox_inches='tight', dpi=200)
    print(f"Plot saved to: {output_img}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str, help="Path to the centerline CSV file")
    args = parser.parse_args()
    plot_centerline_with_map(args.csv_path)
