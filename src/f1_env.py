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
        
        # アクション空間: [ステアリング, 速度] の2次元
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            shape=(2,), 
            dtype=np.float32
        )
        
        # 観測空間: LiDARスキャンデータ（1080次元）
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=30, 
            shape=(1080,), 
            dtype=np.float32
        )

    def reset(self):
        """
        環境をリセットし、初期観測を返す
        
        Returns:
            初期LiDARスキャンデータ
        """
        # configから初期位置を読み込む
        sx, sy, syaw = config.START_POSE
        initial_poses = np.array([[sx, sy, syaw]])
        
        result = self.env.reset(poses=initial_poses)
        obs = result[0] if isinstance(result, tuple) else result
        return obs['scans'][0].astype(np.float32)

    def step(self, action):
        """
        1ステップ実行
        
        Args:
            action: [ステアリング, 速度] の2次元配列
            
        Returns:
            observation: LiDARスキャンデータ
            reward: 報酬
            done: エピソード終了フラグ
            info: 追加情報
        """
        # アクションをシミュレータの入力形式に変換
        steer = action[0] * config.STEER_SENSITIVITY
        speed = config.MIN_SPEED + (action[1] + 1.0) * (config.MAX_SPEED - config.MIN_SPEED) / 2.0
        
        # シミュレータでステップ実行
        obs, _, done, info = self.env.step(np.array([[steer, speed]]))
        scans = obs['scans'][0]
        
        # 報酬計算
        if info is None:
            info = {}
        reward = config.calculate_reward(scans, action, done, speed)
        
        return scans.astype(np.float32), float(reward), bool(done), info
