from PIL import Image
import numpy as np
import shutil
import os

map_path = "my_maps/testmap-tamoku/map-tamoku.pgm"

if not os.path.exists(map_path):
    print(f"Error: {map_path} not found.")
    exit(1)

# バックアップ作成
shutil.copy(map_path, map_path + ".bak")

# 画像を読み込み
img = Image.open(map_path).convert('L')
arr = np.array(img)

# 2値化: 白（255）に近いピクセル以外はすべて黒（0）にする
# これで薄いグレーのノイズ（見えない壁）を完全に消去します
threshold = 200
binary_arr = np.where(arr > threshold, 255, 0).astype(np.uint8)

out_img = Image.fromarray(binary_arr)
out_img.save(map_path)

print(f"Binarized {map_path} successfully. Backup saved as {map_path}.bak")
