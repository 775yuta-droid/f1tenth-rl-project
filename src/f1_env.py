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
        
        # side_distances を正しく計算 (車体端までの距離を考慮し、TTC衝突判定を適正化)
        dist_sides = config.CAR_WIDTH / 2.
        dist_fr = config.CAR_LENGTH / 2.
        RaceCar.side_distances = np.zeros(new_num_beams)
        for i in range(new_num_beams):
            angle = RaceCar.scan_angles[i]
            if abs(angle) < 1e-6: # 真前
                RaceCar.side_distances[i] = dist_fr
            else:
                # 矩形車体の境界までの距離を計算 (minプーリング)
                to_side = dist_sides / abs(np.sin(angle))
                to_fr = dist_fr / abs(np.cos(angle))
                RaceCar.side_distances[i] = min(to_side, to_fr)

        # マシン寸法の適用（config.py の値を物理エンジンに強制）
        self.env.params['length'] = config.CAR_LENGTH
        self.env.params['width'] = config.CAR_WIDTH
        
        # 内部シミュレータとエージェントのパラメータも同期 (幾何学的衝突判定に必須)
        if hasattr(self.env, 'sim'):
            self.env.sim.params['length'] = config.CAR_LENGTH
            self.env.sim.params['width'] = config.CAR_WIDTH
            for agent in self.env.sim.agents:
                agent.params['length'] = config.CAR_LENGTH
                agent.params['width'] = config.CAR_WIDTH
                agent.num_beams = new_num_beams # インスタンス側も更新
            
            self.steer_limit = self.env.sim.agents[0].params['s_max']
        else:
            self.steer_limit = 0.4189
        
        # 観測空間の計算
        # 1. LiDAR: config.LIDAR_BEAMS (1440) -> 270°(1080点) に制限しダウンサンプリング
        self.lidar_size = 1080 // config.LIDAR_DOWNSAMPLE_FACTOR
        
        # 2. 車両状態: [速度, ステアリング] (2次元)
        self.state_size = 2 if config.INCLUDE_VEHICLE_STATE else 0
        
        # 3. LiDAR残差: 現在と前ステップの差分 (同次元)
        self.residual_size = self.lidar_size if config.INCLUDE_LIDAR_RESIDUAL else 0
        
        # 4. 追加スカラー特徴: [front_dist, min_dist] (2次元)
        self.extra_size = 2 if config.INCLUDE_EXTRA_FEATURES else 0
        
        total_obs_size = self.lidar_size + self.residual_size + self.state_size + self.extra_size
        
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
        
        # 観測空間の定義
        # LiDAR成分: 0-1反転正規化後 [0.0, 1.0]
        # 車両状態・追加特徴: 概ね [-3.0, 3.0] 内に収まるため low/high を展開
        self.observation_space = gym.spaces.Box(
            low=-3.0,
            high=3.0,
            shape=(total_obs_size * config.FRAME_STACK,),
            dtype=np.float32
        )

    def _get_obs(self, raw_scans):
        """
        加工済みのLiDARデータを受け取り、残りの前処理（ダウンサンプリング、正規化、積層）を行って返す

        正規化方式 (development-plan.md 推奨):
            lidar_norm = 1.0 - lidar / LIDAR_MAX_RANGE
            -> 近い壁=1.0, 遠い空間=0.0 (寄り辺り感觓が強いほど大きな値)
        """
        # 1440点の中から、正面を中心とした270°（1080点）をスライス
        # 中心(720) ± 540 = 180 〜 1260
        scans = raw_scans[180:1260]
        
        # ダウンサンプリング (minプールで最小距離を保存)
        downsampled = scans.reshape(self.lidar_size, config.LIDAR_DOWNSAMPLE_FACTOR).min(axis=1)
        
        # ΔLiDAR（残差）の計算
        delta_lidar = downsampled - self.prev_lidar
        
        # 現在値を次ステップの「前値」として保存
        self.prev_lidar = downsampled.copy()

        # ==========================================================
        # 0-1 反転正規化 (development-plan.md 推奨方式)
        # NaN/inf は reset/step 側でクリーニング済み。
        # 近い壁 → 1.0, 遠い空間 → 0.0
        # ==========================================================
        lidar_norm = 1.0 - np.clip(downsampled, 0.0, config.LIDAR_MAX_RANGE) / config.LIDAR_MAX_RANGE

        # ※ downsampled は 216点 (1080 / LIDAR_DOWNSAMPLE_FACTOR=5)
        # 1点 = 1.25° (270° / 216点)
        # 中心(正面) = 108点
        _H = downsampled                       # 全体が270°(Hokuyo有効帯域)
        _front = _H[76:140]                    # 前方±40°: 108 ± (40/1.25) = 108 ± 32
        front_raw = float(np.min(_front)) if len(_front) > 0 else config.LIDAR_MAX_RANGE
        min_raw   = float(np.min(_H))     if len(_H) > 0   else config.LIDAR_MAX_RANGE
        # 0-1 スケーリング (近いほど大きな値)
        front_feat = 1.0 - np.clip(front_raw, 0.0, config.LIDAR_MAX_RANGE) / config.LIDAR_MAX_RANGE
        min_feat   = 1.0 - np.clip(min_raw,   0.0, config.LIDAR_MAX_RANGE) / config.LIDAR_MAX_RANGE

        norm_parts = [lidar_norm]
        
        if config.INCLUDE_LIDAR_RESIDUAL:
            # 残差も max_range で正規化 ([-1, 1] 範囲)
            delta_norm = np.clip(delta_lidar / config.LIDAR_MAX_RANGE, -1.0, 1.0)
            norm_parts.append(delta_norm)

        if config.INCLUDE_VEHICLE_STATE:
            # 現在の車両状態を取得 [速度, ステアリング]
            state = self.env.sim.agents[0].state
            # state自体がNaNになる場合があるためガード
            state = np.nan_to_num(state, nan=0.0)
            vel   = float(state[3]) / config.MAX_SPEED       # [0, 1]付近
            steer = float(state[2]) / self.steer_limit       # [-1, 1]
            norm_parts.append(np.array([vel, steer], dtype=np.float32))

        if config.INCLUDE_EXTRA_FEATURES:
            norm_parts.append(np.array([front_feat, min_feat], dtype=np.float32))

        current_obs = np.concatenate(norm_parts).astype(np.float32)

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
        return np.nan_to_num(obs_final, nan=0.0, posinf=1.0, neginf=-1.0)

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
        # 狭路マップ (tamoku) での即死を避けるため、ノイズを ±0.1m -> ±0.02m へ縮小
        sx += np.random.uniform(-0.02, 0.02)
        sy += np.random.uniform(-0.02, 0.02)
        syaw += np.random.uniform(-0.01, 0.01)
        
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
        1ステップ実行 (Action Repeat 導入版)
        """
        # アクションのスケーリング
        # アクションのNaNチェック (保険)
        action = np.nan_to_num(action, nan=0.0)

        # steer: [-1, 1] -> [-max_steer, max_steer] (ラジアン)
        steer = float(action[0]) * self.steer_limit
        # speed: [-1, 1] -> [MIN_SPEED, MAX_SPEED]
        speed = config.MIN_SPEED + (float(action[1]) + 1.0) * (config.MAX_SPEED - config.MIN_SPEED) / 2.0
        
        total_reward = 0.0
        done = False
        info = {}
        clean_scans = None

        # --- Action Repeat ループ ---
        # AIの1回の判断を複数ステップ継続させる
        for _ in range(config.ACTION_REPEAT):
            obs, _, done, info = self.env.step(np.array([[steer, speed]]))
            raw_scans = obs['scans'][0]

            # LiDAR異常値のクリーニング
            clean_scans = np.nan_to_num(raw_scans, nan=30.0, posinf=30.0, neginf=0.0)
            clean_scans = np.clip(clean_scans, 0.0, 30.0)

            # --- EXP-43: サブステップごとに観測バッファを更新 ---
            # これにより FRAME_STACK が「直近の微細な動き」を保持できるようになり、
            # CNNが速度や壁の接近をより正確に抽出可能になる。
            processed_obs = self._get_obs(clean_scans)

            # 現在位置を取得
            state = self.env.sim.agents[0].state
            cur_x, cur_y = state[0], state[1]

            # 報酬計算 (毎サブステップ計算し累積)
            if info is None:
                info = {}
            info['raw_scan'] = clean_scans
            step_reward = calculate_reward(clean_scans, action, done, speed, self.prev_x, self.prev_y, cur_x, cur_y)
            total_reward += step_reward

            # 前位置を更新
            self.prev_x = cur_x
            self.prev_y = cur_y

            if done:
                break

        # 最終サブステップでの加工済み観測（あるいはループ内最後の更新値）を返す
        reward_final = np.nan_to_num(float(total_reward), nan=-1.0)
        
        return processed_obs, reward_final, bool(done), info

