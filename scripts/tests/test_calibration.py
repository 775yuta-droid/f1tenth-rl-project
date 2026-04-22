import sys
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src import config
from src.f1_env import F1TenthRL

# データ収集時は一時的に正規化を無効化
config.NORMALIZE_OBSERVATIONS = False

def calibrate():
    # config.MAP_PATH をそのまま使用
    print(f"Using map: {config.MAP_PATH}")
    env = F1TenthRL(config.MAP_PATH)
    
    lidar_data = []
    residual_data = []
    state_data = []
    
    print("Collecting 10000 steps of data for calibration...")
    # Seed
    np.random.seed(42)
    obs = env.reset()
    
    # 1フレーム分のサイズを算出
    total_obs_size = env.observation_space.shape[0] // config.FRAME_STACK
    lidar_size = env.lidar_size
    state_size = env.state_size
    
    for i in range(10000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        # 積層されたデータから最新の1フレーム分のみを抽出
        # (f1_env.py の _get_obs 実装により、最初の [0:total_obs_size] が最新)
        latest_obs = obs[:total_obs_size]
        
        # 観測データから各要素を抽出
        lidar = latest_obs[:lidar_size]
        lidar_data.append(lidar)
        
        idx = lidar_size
        if config.INCLUDE_LIDAR_RESIDUAL:
            residual = latest_obs[idx:idx+lidar_size]
            residual_data.append(residual)
            idx += lidar_size
            
        if config.INCLUDE_VEHICLE_STATE:
            state = latest_obs[idx:idx+state_size]
            state_data.append(state)
            
        if done:
            env.reset()
            
    print("Done collecting data. Calculating statistics...")
    
    print("\n--- Calibration Results (Update config.py with these) ---")
    
    # LiDAR
    all_lidar = np.array(lidar_data)
    l_mean = np.mean(all_lidar)
    l_std = np.std(all_lidar)
    print(f"LIDAR_MEAN = {l_mean:.3f}")
    print(f"LIDAR_STD = {l_std:.3f}")
    
    # 残差
    if config.INCLUDE_LIDAR_RESIDUAL and len(residual_data) > 0:
        all_res = np.array(residual_data)
        r_mean = np.mean(all_res)
        r_std = np.std(all_res)
        print(f"LIDAR_RESIDUAL_MEAN = {r_mean:.3f}")
        print(f"LIDAR_RESIDUAL_STD = {r_std:.3f}")
    else:
        print("LIDAR_RESIDUAL = (Disabled in config)")
    
    # 車両状態
    if config.INCLUDE_VEHICLE_STATE and len(state_data) > 0:
        all_state = np.array(state_data)
        s_mean = np.mean(all_state, axis=0)
        s_std = np.std(all_state, axis=0)
        # 標準偏差が0になるのを防ぐ
        s_std[s_std < 1e-4] = 1.0
        
        mean_str = ", ".join([f"{m:.3f}" for m in s_mean])
        std_str = ", ".join([f"{s:.3f}" for s in s_std])
        print(f"VEHICLE_STATE_MEAN = np.array([{mean_str}])")
        print(f"VEHICLE_STATE_STD = np.array([{std_str}])")
    else:
        print("VEHICLE_STATE = (Disabled in config)")
        
    print("---------------------------------------------------------")

if __name__ == '__main__':
    calibrate()
