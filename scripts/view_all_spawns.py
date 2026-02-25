"""
全スポーン位置を一括表示するスクリプト

使い方（ホスト環境で実行）:
    python3 scripts/view_all_spawns.py
"""
import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# config をインポート
sys.path.append(os.path.join(os.path.dirname(__file__)))
import config

# ---- マップ読み込み ----
map_yaml = config.MAP_PATH + ".yaml"
with open(map_yaml, 'r') as f:
    map_conf = yaml.safe_load(f)

origin     = map_conf['origin']      # [x_m, y_m, theta]
resolution = map_conf['resolution']  # m/pixel
img_name   = map_conf['image']
img_path   = os.path.join(os.path.dirname(config.MAP_PATH), img_name)

img = Image.open(img_path)
map_img = np.array(img)
height, width = map_img.shape


def world_to_pixel(x, y):
    """ワールド座標 → ピクセル座標"""
    px = (x - origin[0]) / resolution
    py = height - (y - origin[1]) / resolution
    return px, py


# ---- 描画 ----
aspect = width / height
fig, ax = plt.subplots(figsize=(10 * aspect, 10), facecolor='#111')
ax.imshow(map_img, cmap='gray', origin='upper')
ax.axis('off')
ax.set_title('全スポーン位置の確認', color='white', fontsize=14)
fig.patch.set_facecolor('#111')

all_poses = config.START_POSES if config.START_POSE_RANDOMIZE else [config.START_POSE]

colors = plt.cm.hsv(np.linspace(0, 0.9, len(all_poses)))
arrow_len = 8   # ピクセル単位の矢印長さ

legend_patches = []
for i, pose in enumerate(all_poses):
    x, y, yaw = pose
    px, py = world_to_pixel(x, y)
    c = colors[i]

    # 点
    ax.plot(px, py, 'o', color=c, markersize=10, markeredgecolor='white', zorder=5)

    # 向きの矢印
    dx =  arrow_len * np.cos(yaw)
    dy = -arrow_len * np.sin(yaw)  # 画像 y 軸は反転
    ax.arrow(px, py, dx, dy,
             head_width=4, head_length=5,
             fc=c, ec='white', zorder=6)

    # ラベル（座標 & yaw）
    ax.text(px + 3, py - 12,
            f"#{i}  ({x:.1f},{y:.1f})  yaw={yaw:.2f}",
            color=c, fontsize=9, fontfamily='monospace')

    legend_patches.append(
        mpatches.Patch(color=c, label=f"#{i}: ({x},{y}) yaw={yaw:.2f}")
    )

# デフォルトスポーン（単体）もチェック
def_x, def_y, def_yaw = config.START_POSE
def_px, def_py = world_to_pixel(def_x, def_y)
ax.plot(def_px, def_py, '*', color='yellow', markersize=14, zorder=7, label='DEFAULT')

ax.legend(handles=legend_patches, loc='lower right', fontsize=8,
          facecolor='#222', edgecolor='white', labelcolor='white')

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'gif', 'spawn_map.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"保存完了: {out_path}")
plt.show()
