import sys
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src import config
from src.f1_env import F1TenthRL

env = F1TenthRL(config.MAP_PATH)
config.START_POSE_RANDOMIZE = False
config.START_POSE = [10.40, -4.38, 0.0]
obs = env.reset()

# 意図的に「ハンドルを切らない」「ゆっくり直進する」アクションを与える
action = np.array([0.0, -0.5]) # steer=0.0, speed は config.MIN_SPEED と MAX_SPEED の中間より少し遅め

for i in range(5):
    obs, reward, done, info = env.step(action)
    state = env.env.sim.agents[0].state
    min_dist = np.min(obs)
    print(f"Step {i+1}: Speed={state[3]:.2f}, Steer={state[2]:.2f}, X={state[0]:.3f}, Y={state[1]:.3f}, MinScan={min_dist:.3f}, Reward={reward:.2f}")
    if done:
        print(f"-> {i+1} ステップ目で衝突しました！")
        break
else:
    print("-> 20ステップ生存しました!マップと物理エンジンは正常です。")

env.close()
