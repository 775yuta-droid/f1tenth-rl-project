import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

import config
# データ収集時は一時的に正規化を無効化
config.NORMALIZE_OBSERVATIONS = False
config.MAP_PATH = '/opt/f1tenth_gym/gym/f110_gym/envs/maps/levine' # default map for safety

from src.f1_env import F1TenthRL

def calibrate():
    env = F1TenthRL(config.MAP_PATH)
    
    lidar_data = []
    residual_data = []
    state_data = []
    
    print("Collecting 10000 steps of data for calibration...")
    # Seed
    np.random.seed(42)
    obs = env.reset()
    
    lidar_size = env.lidar_size
    
    for i in range(10000):
        # AIのアクション空間に合わせたランダムアクション
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        # 観測データから各要素を抽出
        lidar = obs[:lidar_size]
        lidar_data.append(lidar)
        
        idx = lidar_size
        if config.INCLUDE_LIDAR_RESIDUAL:
            residual = obs[idx:idx+lidar_size]
            residual_data.append(residual)
            idx += lidar_size
            
        if config.INCLUDE_VEHICLE_STATE:
            state = obs[idx:idx+2]
            state_data.append(state)
            
        if done:
            env.reset()
            
    print("Done collecting data. Calculating statistics...")
    
    # 統計情報の計算
    all_lidar = np.array(lidar_data)
    # LiDARは全ピクセル・全ステップでの平均/標準偏差とする
    l_mean = np.mean(all_lidar)
    l_std = np.std(all_lidar)
    
    all_res = np.array(residual_data)
    # 残差も全体での平均/標準偏差
    r_mean = np.mean(all_res)
    r_std = np.std(all_res)
    
    all_state = np.array(state_data)
    # 状態（速度、ステアリング）は各次元ごとに計算
    s_mean = np.mean(all_state, axis=0)
    s_std = np.std(all_state, axis=0)
    # 標準偏差が0になるのを防ぐ
    s_std[s_std < 1e-4] = 1.0
    
    print("\n--- Calibration Results (Update config.py with these) ---")
    print(f"LIDAR_MEAN = {l_mean:.3f}")
    print(f"LIDAR_STD = {l_std:.3f}")
    print(f"LIDAR_RESIDUAL_MEAN = {r_mean:.3f}")
    print(f"LIDAR_RESIDUAL_STD = {r_std:.3f}")
    print(f"VEHICLE_STATE_MEAN = np.array([{s_mean[0]:.3f}, {s_mean[1]:.3f}])  # [vel, steer]")
    print(f"VEHICLE_STATE_STD = np.array([{s_std[0]:.3f}, {s_std[1]:.3f}])")
    print("---------------------------------------------------------")

if __name__ == '__main__':
    calibrate()
