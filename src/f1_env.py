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
from collections import deque

from . import config
from .rewards import calculate_reward


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
        self.env = gym.make('f110-v0', map=map_path, map_ext='.pgm', num_agents=1, timestep=config.SIM_TIMESTEP)
        
        # --- シミュレータの解像度アップグレード (1080 -> 1440本) ---
        # ライブラリ側の制限により gym.make で指定できないため、内部クラス属性を直接書き換える
        from f110_gym.envs.base_classes import RaceCar
        from f110_gym.envs.laser_models import ScanSimulator2D
        
        new_num_beams = config.LIDAR_BEAMS # 1440
        fov = 4.7 # 270度以上の視野 (約269.3度)
        
        # クラス属性の更新 (これによりすべてのエージェントに適用される)
        base_env = self.env.unwrapped
        RaceCar.scan_simulator = ScanSimulator2D(new_num_beams, fov)
        RaceCar.scan_simulator.set_map(base_env.map_path, base_env.map_ext)
        RaceCar.cosines = np.cos(np.linspace(-fov/2., fov/2., new_num_beams))
        RaceCar.scan_angles = np.linspace(-fov/2., fov/2., new_num_beams)
        RaceCar.side_distances = np.ones(new_num_beams)

        # マシン寸法の適用（config.py の値を物理エンジンに強制）
        self.env.params['length'] = config.CAR_LENGTH
        self.env.params['width'] = config.CAR_WIDTH
        
        # 内部エージェントのパラメータをより確実に反映
        if hasattr(self.env, 'sim') and len(self.env.sim.agents) > 0:
            agent = self.env.sim.agents[0]
            agent.params.update({
                'length': config.CAR_LENGTH,
                'width': config.CAR_WIDTH
            })
            # 属性としても保持されている可能性があるため上書き
            if hasattr(agent, 'length'): agent.length = config.CAR_LENGTH
            if hasattr(agent, 'width'): agent.width = config.CAR_WIDTH
            
            agent.num_beams = new_num_beams
            self.steer_limit = agent.params.get('s_max', 0.4189)
            
            # デバッグ用：実際に適用されているサイズを表示
            print(f"[DEBUG] Sim Width: {agent.params['width']}")
            print(f"[DEBUG] Sim Length: {agent.params['length']}")
        else:
            self.steer_limit = 0.4189
        
        # 観測空間の計算
        # 1. LiDAR: config.LIDAR_BEAMS (1440) -> 270°(1080点) に制限しダウンサンプリング
        self.lidar_size = 1080 // config.LIDAR_DOWNSAMPLE_FACTOR
        
        # 2. 車両状態: [速度, ステアリング] (2次元)
        self.state_size = 2 if config.INCLUDE_VEHICLE_STATE else 0
        
        # 3. LiDAR残差: 現在と前ステップの差分 (同次元)
        self.residual_size = self.lidar_size if config.INCLUDE_LIDAR_RESIDUAL else 0
        
        total_obs_size = self.lidar_size + self.residual_size + self.state_size
        
        # 前ステップのLiDAR（Δ=0で初期化）
        self.prev_lidar = np.zeros(self.lidar_size, dtype=np.float32)

        # 前ステップの車両位置（走行距離報酬用）
        self.prev_x = 0.0
        self.prev_y = 0.0
        
        # フレーム積層用バッファ (間引きを考慮したサイズ)
        self.obs_buffer = deque(maxlen=(config.FRAME_STACK - 1) * config.FRAME_SKIP + 1)
        
        # 現在のステアリング角 (EXP-20: Delta制御用)
        self.current_steer = 0.0
        
        # アクション空間: [ステアリング, 速度] の2次元
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            shape=(2,), 
            dtype=np.float32
        )
        
        # 観測空間の定義 (FRAME_STACK分を結合するため次元数を拡大)
        self.observation_space = gym.spaces.Box(
            low=-30, 
            high=30, 
            shape=(total_obs_size * config.FRAME_STACK,), 
            dtype=np.float32
        )

    def _get_obs(self, raw_scans):
        """
        加工済みのLiDARデータを受け取り、残りの前処理（ダウンサンプリング、正規化、積層）を行って返す
        """
        # 1440点の中から、正面を中心とした270°（1080点）をスライス
        # 中心(720) ± 540 = 180 〜 1260
        scans = raw_scans[180:1260]
        
        # ダウンサンプリング
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

        if config.NORMALIZE_OBSERVATIONS:
            norm_parts = []
            # LiDAR 正規化
            lidar_norm = (downsampled - config.LIDAR_MEAN) / config.LIDAR_STD
            norm_parts.append(lidar_norm)
            if config.INCLUDE_LIDAR_RESIDUAL:
                delta_norm = (delta_lidar - config.LIDAR_RESIDUAL_MEAN) / config.LIDAR_RESIDUAL_STD
                norm_parts.append(delta_norm)
            if config.INCLUDE_VEHICLE_STATE:
                state_arr = np.array([vel, steer], dtype=np.float32)
                state_norm = (state_arr - config.VEHICLE_STATE_MEAN) / config.VEHICLE_STATE_STD
                norm_parts.append(state_norm)
            current_obs = np.concatenate(norm_parts).astype(np.float32)
        else:
            current_obs = np.concatenate(parts).astype(np.float32)

        # フレーム積層処理 (間引きを適用)
        self.obs_buffer.append(current_obs)
        
        # バッファから指定間隔でフレームを抽出
        stacked_obs = []
        for i in range(config.FRAME_STACK):
            idx = -(i * config.FRAME_SKIP + 1)
            if abs(idx) > len(self.obs_buffer):
                stacked_obs.append(self.obs_buffer[0])
            else:
                stacked_obs.append(self.obs_buffer[idx])
        
        # 最終出力のNaNチェック
        obs_final = np.concatenate(stacked_obs)
        return np.nan_to_num(obs_final, nan=0.0)

    def reset(self):
        """
        環境をリセットし、初期観測を返す
        """
        # スタート位置の選択（ランダム化 or 固定）
        if config.START_POSE_RANDOMIZE and len(config.START_POSES) > 0:
            pose = config.START_POSES[np.random.randint(len(config.START_POSES))]
        else:
            pose = config.START_POSE
        sx, sy, syaw = pose
        
        # EXP-19: スタート位置にノイズを付加 (丸暗記防止)
        sx += np.random.uniform(-0.1, 0.1)
        sy += np.random.uniform(-0.1, 0.1)
        syaw += np.random.uniform(-0.05, 0.05)
        
        initial_poses = np.array([[sx, sy, syaw]])

        result = self.env.reset(poses=initial_poses)
        raw_obs = result[0] if isinstance(result, tuple) else result
        raw_scans = raw_obs['scans'][0]
        # LiDAR異常値のクリーニング (報酬計算・観測の両方で共有)
        clean_scans = np.nan_to_num(raw_scans, nan=30.0, posinf=30.0, neginf=0.0)
        clean_scans = np.clip(clean_scans, 0.0, 30.0)

        # バッファのリセット
        self.obs_buffer.clear()

        return self._get_obs(clean_scans)

    def step(self, action):
        """
        1ステップ実行
        """
        # アクションのスケーリング
        # steer: [-1, 1] -> [-max_steer, max_steer] (ラジアン)
        steer = float(action[0]) * self.steer_limit
        # speed: [-1, 1] -> [MIN_SPEED, MAX_SPEED]
        speed = config.MIN_SPEED + (float(action[1]) + 1.0) * (config.MAX_SPEED - config.MIN_SPEED) / 2.0
        
        # シミュレーション実行
        obs, _, done, info = self.env.step(np.array([[steer, speed]]))
        raw_scans = obs['scans'][0]

        # LiDAR異常値のクリーニング (報酬計算・観測の両方で共有)
        clean_scans = np.nan_to_num(raw_scans, nan=30.0, posinf=30.0, neginf=0.0)
        clean_scans = np.clip(clean_scans, 0.0, 30.0)

        # 現在位置を取得
        state = self.env.sim.agents[0].state
        cur_x, cur_y = state[0], state[1]

        # 報酬計算
        if info is None:
            info = {}
        info['raw_scan'] = clean_scans
        reward = calculate_reward(clean_scans, action, done, speed, self.prev_x, self.prev_y, cur_x, cur_y)

        # 前位置を更新
        self.prev_x = cur_x
        self.prev_y = cur_y

        processed_obs = self._get_obs(clean_scans)

        # 最終出力のNaNチェック (保険)
        reward_final = np.nan_to_num(float(reward), nan=-1.0)
        
        return processed_obs, reward_final, bool(done), info


