"""
F1Tenth Gym Environment Wrapper for Reinforcement Learning

このモジュールは、F1Tenth Gymシミュレータをラップし、
Stable Baselines3で使用可能なGym環境を提供します。
"""

import gym
import f110_gym
import numpy as np
import sys
import os

# scriptsディレクトリからconfigをimportできるようにパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import config


class F1TenthRL(gym.Env):
    """
    F1Tenth強化学習環境クラス
    
    LiDARセンサーデータ（1080次元）を観測空間とし、
    ステアリングと速度の2次元連続アクションを出力します。
    """
    
    def __init__(self, map_path: str):
        """
        Args:
            map_path: マップファイルのパス（拡張子なし）
        """
        super(F1TenthRL, self).__init__()
        self.env = gym.make('f110-v0', map=map_path, map_ext='.pgm', num_agents=1)
        
        # 観測空間の計算
        # 1. LiDAR: 1080 -> ダウンサンプリング
        self.lidar_size = 1080 // config.LIDAR_DOWNSAMPLE_FACTOR
        
        # 2. 車両状態: [速度, ステアリング] (2次元)
        self.state_size = 2 if config.INCLUDE_VEHICLE_STATE else 0
        
        # 3. LiDAR残差: 現在と前ステップの差分 (同次元)
        self.residual_size = self.lidar_size if config.INCLUDE_LIDAR_RESIDUAL else 0
        
        total_obs_size = self.lidar_size + self.residual_size + self.state_size
        
        # 前ステップのLiDAR（Δ=0で初期化）
        self.prev_lidar = np.zeros(self.lidar_size, dtype=np.float32)
        
        # アクション空間: [ステアリング, 速度] の2次元
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            shape=(2,), 
            dtype=np.float32
        )
        
        # 観測空間の定義
        self.observation_space = gym.spaces.Box(
            low=-30, 
            high=30, 
            shape=(total_obs_size,), 
            dtype=np.float32
        )

    def _get_obs(self, raw_obs):
        """
        生の観測データを加工して返す
        """
        # LiDARデータのダウンサンプリング (平均または最小値を取る)
        scans = raw_obs['scans'][0]
        downsampled = scans.reshape(self.lidar_size, config.LIDAR_DOWNSAMPLE_FACTOR).min(axis=1)
        
        # ΔLiDAR（残差）の計算
        delta_lidar = downsampled - self.prev_lidar
        
        # 現在値を次ステップの「前値」として保存
        self.prev_lidar = downsampled.copy()
        
        parts = [downsampled]
        
        if config.INCLUDE_LIDAR_RESIDUAL:
            parts.append(delta_lidar)
        
        if config.INCLUDE_VEHICLE_STATE:
            # 現在の車両状態を取得 [速度, ステアリング]
            state = self.env.sim.agents[0].state
            vel = state[3] / config.MAX_SPEED
            steer = state[2]
            parts.append(np.array([vel, steer], dtype=np.float32))
        
        return np.concatenate(parts).astype(np.float32)

    def reset(self):
        """
        環境をリセットし、初期観測を返す
        """
        sx, sy, syaw = config.START_POSE
        initial_poses = np.array([[sx, sy, syaw]])
        
        result = self.env.reset(poses=initial_poses)
        raw_obs = result[0] if isinstance(result, tuple) else result
        
        # 初期状態のLiDARを取得してprev_lidarをセット
        scans = raw_obs['scans'][0]
        self.prev_lidar = scans.reshape(self.lidar_size, config.LIDAR_DOWNSAMPLE_FACTOR).min(axis=1)
        
        return self._get_obs(raw_obs)

    def step(self, action):
        """
        1ステップ実行
        """
        steer = action[0] * config.STEER_SENSITIVITY
        speed = config.MIN_SPEED + (action[1] + 1.0) * (config.MAX_SPEED - config.MIN_SPEED) / 2.0
        
        obs, _, done, info = self.env.step(np.array([[steer, speed]]))
        raw_scans = obs['scans'][0]
        
        # 報酬計算 (報酬計算には生のLiDARデータを使う)
        if info is None:
            info = {}
        info['raw_scan'] = raw_scans # 描画用に生のデータを保持
        reward = config.calculate_reward(raw_scans, action, done, speed)
        
        processed_obs = self._get_obs(obs)
        
        return processed_obs, float(reward), bool(done), info

