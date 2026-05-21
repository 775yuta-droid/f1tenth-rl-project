import sys
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src import config
from src.f1_env import F1TenthRL

env = F1TenthRL(config.MAP_PATH)
config.START_POSE_RANDOMIZE = False
config.START_POSE = [7.5, -3.5, 0.0]

obs = env.reset()
print("Spawned at [7.5, -3.5, 0.0]")
action = np.array([1.0, 1.0])

for i in range(3):
    state = env.env.sim.agents[0].state
    x, y, yaw = state[0], state[1], state[4]
    
    print(f"Step {i} before action: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}")
    
    obs, reward, done, info = env.step(action)
    print(f"  -> reward={reward}, done={done}")
    if done:
        break

env.close()
