import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

import config
# Ensure normalization is active
config.NORMALIZE_OBSERVATIONS = True
config.MAP_PATH = '/opt/f1tenth_gym/gym/f110_gym/envs/maps/levine'

from src.f1_env import F1TenthRL

def test_norm():
    env = F1TenthRL(config.MAP_PATH)
    
    np.random.seed(42)
    obs = env.reset()
    
    lidar_size = env.lidar_size
    all_obs = []
    
    print("Collecting 1000 steps with NORMALIZE_OBSERVATIONS = True...")
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        all_obs.append(obs)
        if done:
            env.reset()
            
    all_obs = np.array(all_obs)
    
    # Processed LiDAR
    lidar_norm = all_obs[:, :lidar_size]
    print(f"Normalized LiDAR - Mean: {np.mean(lidar_norm):.3f}, Std: {np.std(lidar_norm):.3f}")
    
    idx = lidar_size
    if config.INCLUDE_LIDAR_RESIDUAL:
        delta_norm = all_obs[:, idx:idx+lidar_size]
        print(f"Normalized Residual - Mean: {np.mean(delta_norm):.3f}, Std: {np.std(delta_norm):.3f}")
        idx += lidar_size
        
    if config.INCLUDE_VEHICLE_STATE:
        state_norm = all_obs[:, idx:idx+2]
        s_mean = np.mean(state_norm, axis=0)
        s_std = np.std(state_norm, axis=0)
        print(f"Normalized State - Mean: [{s_mean[0]:.3f}, {s_mean[1]:.3f}], Std: [{s_std[0]:.3f}, {s_std[1]:.3f}]")
        
    print("\nSUCCESS! If values are approximately Mean: 0.0, Std: 1.0, normalization is working.")

if __name__ == '__main__':
    test_norm()
