import sys
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src import config
from src.f1_env import F1TenthRL

# 環境初期化
env = F1TenthRL(config.MAP_PATH)

poses = [
    [4.2, -0.2, 3.2],
    [-1.8, -0.1, 3.1],
    [7.5, -3.5, 0.0],
    [-2.2, -3.5, 0.0],
]

print("--- 物理エンジンによる初期位置のLiDARスキャン確認 ---")
for i, pose in enumerate(poses):
    config.START_POSE_RANDOMIZE = False
    config.START_POSE = pose
    obs = env.reset()
    
    # f1_env.reset() から生のlidar(1440点)を取得する方法はないので、
    # 最初のstepを踏まずに、infoから生scanを取得するか、シミュレータに直接アクセス
    sim_obs, _, _, _ = env.env.step(np.array([[0.0, 0.0]]))
    raw_scans = sim_obs['scans'][0]
    
    min_dist = np.min(raw_scans)
    front_dist = raw_scans[720] # 正面
    
    print(f"Pose #{i} {pose}: 最小距離={min_dist:.3f}m, 正面距離={front_dist:.3f}m")
    
    if min_dist < 0.2: # 車の幅が0.19なので中心から0.095。余裕を見て0.2以下ならほぼ激突
        print("  -> [警告] 壁に激突しているか、非常に近いです！")

env.close()
