"""
センターライン（ウェイポイント）可視化スクリプト

CSVファイルを読み込み、コースの形状と曲率をプロットします。
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_centerline(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return

    # CSV読み込み (numpyとcsvを使用)
    xs, ys, hs, ks = [], [], [], []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                xs.append(float(row["x"]))
                ys.append(float(row["y"]))
                hs.append(float(row["heading"]))
                ks.append(float(row["curvature"]))
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    x = np.array(xs)
    y = np.array(ys)
    heading = np.array(hs)
    curvature = np.array(ks)

    if len(x) == 0:
        print("No data found in CSV.")
        return

    plt.figure(figsize=(10, 10))

    # 曲率に基づいて色を変えてプロット
    sc = plt.scatter(x, y, c=np.abs(curvature), cmap='jet', s=15, label='Waypoints', zorder=3)
    plt.colorbar(sc, label='Curvature [1/m]')

    # 線でつなぐ
    plt.plot(x, y, 'k-', alpha=0.3, zorder=1)

    # 開始地点と終了地点を強調
    plt.plot(x[0], y[0], 'go', markersize=10, label='Start', zorder=4)
    plt.plot(x[-1], y[-1], 'ro', markersize=10, label='End', zorder=4)

    # ヘディング方向の矢印を一部表示 (データ数に応じて調整)
    step = max(1, len(x) // 20)
    q_idx = range(0, len(x), step)
    plt.quiver(x[q_idx], y[q_idx], 
               np.cos(heading[q_idx]), np.sin(heading[q_idx]),
               color='gray', scale=25, width=0.005, alpha=0.6, label='Heading', zorder=2)

    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title(f'Centerline Visualization: {os.path.basename(csv_path)}')
    plt.legend()

    # 保存
    output_img = csv_path.replace('.csv', '.png')
    plt.savefig(output_img, bbox_inches='tight', dpi=150)
    print(f"Plot saved to: {output_img}")
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centerline Plotter")
    parser.add_argument("csv_path", type=str, help="Path to the centerline CSV file")
    args = parser.parse_args()

    plot_centerline(args.csv_path)
