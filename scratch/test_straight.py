import sys
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src import config
from src.f1_env import F1TenthRL

env = F1TenthRL(config.MAP_PATH)
config.START_POSE_RANDOMIZE = False
config.START_POSE = [1.75, 3.17, 0.0]  # 究極の安全地点 (壁まで 2.75m)
obs = env.reset()

# 意図的に「ハンドルを切らない」「ゆっくり直進する」アクションを与える
action = np.array([0.0, -0.5]) # steer=0.0, speed は config.MIN_SPEED と MAX_SPEED の中間より少し遅め

for i in range(5):
    obs, reward, done, info = env.step(action)
    state = env.env.sim.agents[0].state
    # obs は [lidar(216), state(2), extra(2)] * stack(4)
    current_lidar = obs[:216]
    min_lidar_val = np.max(current_lidar)
    real_min_dist = (1.0 - min_lidar_val) * config.LIDAR_MAX_RANGE
    
    print(f"Step {i+1}: Speed={state[3]:.2f}, Steer={state[2]:.2f}, X={state[0]:.3f}, Y={state[1]:.3f}, MinDist={real_min_dist:.3f}m, Reward={reward:.2f}")
    print(f"      Info: {info}")
    if done:
        print(f"-> {i+1} ステップ目で衝突しました！")
        # 衝突時の生のLiDARスキャン（数点）を確認
        # env.env.sim.agents[0].last_scan を取得
        raw_scan = env.env.sim.agents[0].last_scan
        print(f"      Raw Min Scan: {np.min(raw_scan):.3f}m")
        break
else:
    print("-> 20ステップ生存しました!マップと物理エンジンは正常です。")

env.close()
