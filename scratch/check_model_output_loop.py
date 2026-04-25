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

print("--- 連続実行によるクラッシュテスト ---")
obs = env.reset()

for i in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(f"Step {i+1}: Action={action}, Speed={env.env.sim.agents[0].state[3]:.2f}, Steer={env.env.sim.agents[0].state[2]:.2f}, Reward={reward:.2f}")
    if done:
        print(f"-> {i+1} ステップ目で衝突しました！")
        break

env.close()
