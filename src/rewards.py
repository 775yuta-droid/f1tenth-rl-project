"""
報酬計算モジュール

config.py から分離した報酬ロジックを集約します。
calculate_reward() が唯一のエントリーポイントです。
"""
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import config


def calculate_reward(
    scans,
    action,
    done: bool,
    current_speed: float,
    prev_x: float = 0.0,
    prev_y: float = 0.0,
    cur_x: float = 0.0,
    cur_y: float = 0.0,
) -> float:
    """
    1ステップ分の報酬を計算して返す。

    Args:
        scans: LiDARの距離データ (1080点)
        action: AIの出力 [ステアリング, 速度]
        done: 衝突判定フラグ
        current_speed: 現在の車の速度 (m/s)
        prev_x, prev_y: 前ステップの位置
        cur_x, cur_y: 現在の位置

    Returns:
        float: 報酬値
    """
    if done:
        return config.REWARD_COLLISION

    # 1. 前方空間報酬（視野を±45度相当まで拡大: 350～730番）
    front_dist = np.min(scans[350:730])
    reward = (front_dist / 30.0) * config.REWARD_FRONT_WEIGHT

    # 2. 速度報酬 / コーナー前ペナルティ
    speed_factor = current_speed / config.MAX_SPEED
    if front_dist < 2.0:
        # コーナー直前: 速度を出すとペナルティ
        reward -= speed_factor * config.REWARD_SPEED_WEIGHT * 1.0
        progress_scale = 0.0
    elif front_dist < 4.0:
        reward += speed_factor * config.REWARD_SPEED_WEIGHT * 0.1
        progress_scale = 0.3
    else:
        reward += speed_factor * config.REWARD_SPEED_WEIGHT
        progress_scale = 1.0

    # 3. 壁接近ペナルティ（安全マージン）
    min_dist = np.min(scans)
    if min_dist < 1.0:
        reward -= config.REWARD_DISTANCE_WEIGHT * (1.0 - (min_dist / 1.0))

    # 4. 中央維持報酬 (Lateral Centrality)
    left_dist = np.min(scans[700:740])
    right_dist = np.min(scans[340:380])
    centrality = 1.0 - abs(left_dist - right_dist) / (left_dist + right_dist + 1e-6)
    reward += centrality * config.REWARD_CENTRALITY_WEIGHT

    # 5. 走行距離報酬（円形走行抑制）
    progress = np.sqrt((cur_x - prev_x) ** 2 + (cur_y - prev_y) ** 2)
    reward += progress * config.REWARD_PROGRESS_WEIGHT * progress_scale

    # 6. ステアリング安定性（条件付き）
    if front_dist > 5.0:
        reward += (1.0 - abs(action[0])) * 0.2
    reward += config.REWARD_SURVIVAL

    return reward
