import cv2
import numpy as np
import yaml

map_path = '/workspace/my_maps/testmap-tamoku/map-tamoku.pgm'
yaml_path = '/workspace/my_maps/testmap-tamoku/map-tamoku.yaml'

with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
height, width = img.shape
origin = config['origin']
res = config['resolution']

# 完全に白い（障害物がない）ピクセルを探す
free_pixels = np.where(img == 255)

# 中心付近の安全なピクセルをいくつか選ぶ
print("--- 安全なスポーン地点（ワールド座標） ---")
for i in range(0, len(free_pixels[0]), len(free_pixels[0])//5):
    py = free_pixels[0][i]
    px = free_pixels[1][i]
    
    # F1TenthGym の内部計算と同じ逆変換
    # py = height - (y - origin[1]) / res
    # y - origin[1] = (height - py) * res
    y = origin[1] + (height - py) * res
    x = origin[0] + px * res
    
    print(f"[{x:.2f}, {y:.2f}, 0.0]")
