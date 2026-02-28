"""
報酬計算モジュール

config.py から分離した報酬ロジックを集約します。
calculate_reward() が唯一のエントリーポイントです。

テスト時は RewardConfig を使ってパラメータをモックできます:
    from src.rewards import RewardConfig, calculate_reward
    cfg = RewardConfig(reward_collision=-100.0, ...)
    r = calculate_reward(scans, action, done, speed, reward_config=cfg)
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RewardConfig:
    """報酬計算に必要なハイパーパラメータをまとめた設定クラス。"""
    reward_collision: float = -2000.0
    reward_survival: float = 0.02
    reward_front_weight: float = 3.0
    reward_speed_weight: float = 1.0
    reward_centrality_weight: float = 0.5
    reward_distance_weight: float = 1.0
    reward_progress_weight: float = 2.0
    max_speed: float = 2.5


def _load_default_config() -> RewardConfig:
    """scripts/config.py からデフォルト設定を読み込む。"""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    import config
    return RewardConfig(
        reward_collision=config.REWARD_COLLISION,
        reward_survival=config.REWARD_SURVIVAL,
        reward_front_weight=config.REWARD_FRONT_WEIGHT,
        reward_speed_weight=config.REWARD_SPEED_WEIGHT,
        reward_centrality_weight=config.REWARD_CENTRALITY_WEIGHT,
        reward_distance_weight=config.REWARD_DISTANCE_WEIGHT,
        reward_progress_weight=config.REWARD_PROGRESS_WEIGHT,
        max_speed=config.MAX_SPEED,
    )


def calculate_reward(
    scans,
    action,
    done: bool,
    current_speed: float,
    prev_x: float = 0.0,
    prev_y: float = 0.0,
    cur_x: float = 0.0,
    cur_y: float = 0.0,
    reward_config: RewardConfig = None,
) -> float:
    """
    1ステップ分の報酬を計算して返す。

    Args:
        scans: LiDARの距離データ (1080点)
        action: AIの出力 [ステアリング, 速度]
        done: 衝突判定フラグ (True=衝突)
        current_speed: 現在の車の速度 (m/s)
        prev_x, prev_y: 前ステップの位置
        cur_x, cur_y: 現在の位置
        reward_config: 報酬パラメータ。None の場合は config.py から自動読み込み。
                       テスト時は RewardConfig オブジェクトを渡すことでモック可能。

    Returns:
        float: 報酬値
    """
    cfg = reward_config if reward_config is not None else _load_default_config()

    if done:
        return cfg.reward_collision

    # 1. 前方空間報酬（視野を±45度相当まで拡大: 350～730番）
    front_dist = np.min(scans[350:730])
    reward = (front_dist / 30.0) * cfg.reward_front_weight

    # 2. 速度報酬 / コーナー前ペナルティ
    speed_factor = current_speed / cfg.max_speed
    if front_dist < 2.0:
        reward -= speed_factor * cfg.reward_speed_weight * 1.0
        progress_scale = 0.0
    elif front_dist < 4.0:
        reward += speed_factor * cfg.reward_speed_weight * 0.1
        progress_scale = 0.3
    else:
        reward += speed_factor * cfg.reward_speed_weight
        progress_scale = 1.0

    # 3. 壁接近ペナルティ（安全マージン）
    min_dist = np.min(scans)
    if min_dist < 1.0:
        reward -= cfg.reward_distance_weight * (1.0 - (min_dist / 1.0))

    # 4. 中央維持報酬 (Lateral Centrality)
    left_dist = np.min(scans[700:740])
    right_dist = np.min(scans[340:380])
    centrality = 1.0 - abs(left_dist - right_dist) / (left_dist + right_dist + 1e-6)
    reward += centrality * cfg.reward_centrality_weight

    # 5. 走行距離報酬（円形走行抑制）
    progress = np.sqrt((cur_x - prev_x) ** 2 + (cur_y - prev_y) ** 2)
    reward += progress * cfg.reward_progress_weight * progress_scale

    # 6. ステアリング安定性（条件付き）
    if front_dist > 5.0:
        reward += (1.0 - abs(action[0])) * 0.2
    reward += cfg.reward_survival

    return reward
