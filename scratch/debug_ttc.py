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
action = np.array([0.0, -0.5])

for i in range(3):
    obs, reward, done, info = env.step(action)
    state = env.env.sim.agents[0].state
    vel = state[3]
    
    # get scan and other vars
    scan = env.env.sim.agents[0].scan
    scan_angles = env.env.sim.agents[0].scan_angles
    cosines = env.env.sim.agents[0].cosines
    side_distances = env.env.sim.agents[0].side_distances
    ttc_thresh = env.env.sim.agents[0].ttc_thresh
    
    # compute TTC
    ttcs = np.full_like(scan, np.inf)
    if vel != 0.0:
        proj_vel = vel * cosines
        valid = proj_vel > 0
        ttcs[valid] = (scan[valid] - side_distances[valid]) / proj_vel[valid]
    
    min_ttc_idx = np.argmin(ttcs)
    min_ttc = ttcs[min_ttc_idx]
    
    print(f"Step {i+1}:")
    print(f"  Min TTC: {min_ttc:.5f} (Thresh: {ttc_thresh}) at idx {min_ttc_idx}")
    print(f"  scan: {scan[min_ttc_idx]:.3f}, side_dist: {side_distances[min_ttc_idx]:.3f}, proj_vel: {vel*cosines[min_ttc_idx]:.3f}")
    print(f"  MinDist in scan: {np.min(scan):.3f}")
    if done:
        print(f"-> 衝突しました！")
        break

env.close()
