import sys
import os
import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src import config
from src.f1_env import F1TenthRL

env = F1TenthRL(config.MAP_PATH)
model_path = "/workspace/models/ppo_10M_exp40_cnn_verify.zip"
model = PPO.load(model_path, device="cpu")

print("--- モデルの出力確認 ---")
obs = env.reset()
print(f"Initial obs stats: min={np.min(obs):.3f}, max={np.max(obs):.3f}, has_nan={np.isnan(obs).any()}")

action, _ = model.predict(obs, deterministic=True)
print(f"Action: {action}")

obs, reward, done, info = env.step(action)
print(f"Step 1: reward={reward:.2f}, done={done}, state={env.env.sim.agents[0].state}")
if done:
    print("-> 衝突しました！")

env.close()
