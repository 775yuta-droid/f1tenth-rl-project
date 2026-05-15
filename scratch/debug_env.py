import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.f1_env import F1TenthRL
from src import config

def test_env():
    print("Testing environment...")
    env = F1TenthRL(config.MAP_PATH)
    obs = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: Reward={reward:.4f}, Speed={info.get('actual_speed', 'N/A')}")
        if done:
            env.reset()
    print("Test finished successfully!")

if __name__ == "__main__":
    test_env()
