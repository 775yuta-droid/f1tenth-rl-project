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

state = env.env.sim.agents[0].state
x, y = state[0], state[1]

# F1Tenth uses distance transform dt
dt_val = env.env.sim.dt
# get pixel
px = int((x - env.env.sim.map_origin[0]) / env.env.sim.map_resolution)
py = int((y - env.env.sim.map_origin[1]) / env.env.sim.map_resolution)

print(f"Car Center DT: {dt_val[px, py] if px < dt_val.shape[0] and py < dt_val.shape[1] else 'OOB'}")

