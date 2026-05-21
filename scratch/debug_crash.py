import sys
import os
import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src import config
from src.f1_env import F1TenthRL

env = F1TenthRL(config.MAP_PATH)
model_path = "/workspace/models/ppo_10M_exp25_fast_stable.zip"
model = PPO.load(model_path, device="cpu")

print("--- 衝突デバッグ開始 ---")
for ep in range(5):
    # ランダムではなく順番に試す
    pose = config.START_POSES[ep % len(config.START_POSES)]
    config.START_POSE_RANDOMIZE = False
    config.START_POSE = pose
    
    obs = env.reset()
    
    print(f"Episode {ep}, Spawn: {pose}")
    for step in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        sim_obs, _, _, _ = env.env.step(np.array([[0.0, 0.0]]))
        raw_scans = sim_obs['scans'][0]
        min_dist = np.min(raw_scans)
        
        state = env.env.sim.agents[0].state
        print(f"  Step {step}: Action={action}, Speed={state[3]:.2f}, Steer={state[2]:.2f}, MinDist={min_dist:.3f}")
        
        if done:
            print(f"  --> 衝突発生！ Step {step}")
            break

env.close()
