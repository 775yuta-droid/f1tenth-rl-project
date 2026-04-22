import sys
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src import config
from src.f1_env import F1TenthRL

# Ensure normalization is active
config.NORMALIZE_OBSERVATIONS = True

def test_norm():
    # config.MAP_PATH をそのまま使用
    print(f"Using map: {config.MAP_PATH}")
    env = F1TenthRL(config.MAP_PATH)
    
    np.random.seed(42)
    obs = env.reset()
    
    lidar_size = env.lidar_size
    state_size = env.state_size
    total_obs_size = env.observation_space.shape[0] // config.FRAME_STACK
    
    all_obs = []
    
    print(f"Collecting 1000 steps with NORMALIZE_OBSERVATIONS = True (FrameStack: {config.FRAME_STACK})...")
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        all_obs.append(obs)
        if done:
            env.reset()
            
    all_obs = np.array(all_obs)
    
    # 統計情報のチェック（スタックされた全フレームを含めてチェック）
    print("\n--- Normalization Test Results (Across all stacked frames) ---")
    
    # 全データのうち、1フレームあたりの各要素のインデックスを特定
    for f in range(config.FRAME_STACK):
        offset = f * total_obs_size
        print(f"\n[Frame {f}] (Offset: {offset})")
        
        # Processed LiDAR
        lidar_norm = all_obs[:, offset : offset + lidar_size]
        l_mean, l_std = np.mean(lidar_norm), np.std(lidar_norm)
        print(f"  LiDAR    - Mean: {l_mean:7.3f}, Std: {l_std:7.3f} {'[OK]' if np.abs(l_mean) < 0.2 and np.abs(l_std-1.0) < 0.2 else '[??]'}")
        
        idx = offset + lidar_size
        if config.INCLUDE_LIDAR_RESIDUAL:
            if np.isnan(config.LIDAR_RESIDUAL_MEAN):
                print("  Residual - [SKIP] config values are NaN")
            else:
                delta_norm = all_obs[:, idx : idx + lidar_size]
                r_mean, r_std = np.mean(delta_norm), np.std(delta_norm)
                print(f"  Residual - Mean: {r_mean:7.3f}, Std: {r_std:7.3f} {'[OK]' if np.abs(r_mean) < 0.2 and np.abs(r_std-1.0) < 0.2 else '[??]'}")
            idx += lidar_size
            
        if config.INCLUDE_VEHICLE_STATE:
            state_norm = all_obs[:, idx : idx + state_size]
            s_mean = np.mean(state_norm, axis=0)
            s_std = np.std(state_norm, axis=0)
            print(f"  State    - Mean: {s_mean}, Std: {s_std}")
        
    print("\nVerification complete. If values are approximately Mean: 0.0, Std: 1.0, normalization is working correctly.")

if __name__ == '__main__':
    test_norm()
