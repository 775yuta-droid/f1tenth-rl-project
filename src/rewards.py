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
from src import config


@dataclass
class RewardConfig:
    """報酬計算に必要なハイパーパラメータをまとめた設定クラス。"""
    reward_collision: float = -1000.0
    reward_survival: float = 0.05
    reward_front_weight: float = 3.0
    reward_speed_weight: float = 1.0
    reward_safety_weight: float = 0.8  # 旧 centrality_weight + distance_weight を統合
    reward_distance_weight: float = 1.0   # 互換性のため残存（safety_weight が主役）
    reward_progress_weight: float = 1.0
    max_speed: float = 2.5


def _load_default_config() -> RewardConfig:
    """src/config.py からデフォルト設定を読み込む。"""
    return RewardConfig(
        reward_collision=config.REWARD_COLLISION,
        reward_survival=config.REWARD_SURVIVAL,
        reward_front_weight=config.REWARD_FRONT_WEIGHT,
        reward_speed_weight=config.REWARD_SPEED_WEIGHT,
        reward_safety_weight=config.REWARD_SAFETY_WEIGHT,
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

    # 1. 前方空間報酬（視野を±60度相当: 180～900番に拡大、斜め突進の抑制）
    front_dist = np.min(scans[180:900])
    reward = (front_dist / 30.0) * cfg.reward_front_weight

    # 2. 速度報酬 / コーナー前ペナルティ
    speed_factor = current_speed / cfg.max_speed
    if front_dist < 2.0:
        reward -= speed_factor * cfg.reward_speed_weight * 1.0
        progress_scale = 0.1  # 完全に停止しないよう下限を設定
    elif front_dist < 4.0:
        reward += speed_factor * cfg.reward_speed_weight * 0.1
        progress_scale = 0.3
    else:
        reward += speed_factor * cfg.reward_speed_weight
        progress_scale = 1.0

    # 3+4. 安全距離スコア（壁ペナルティ＋クリアランス報酬を統合）
    # 2m 以上: プラス報酬 (最大 +safety_weight*0.5)
    # 0m:     マイナスペナルティ (最大 -safety_weight*0.5)
    # 連続関数なので学習勾配が滑らか
    wall_dist = np.min(scans)
    safety_score = np.clip(wall_dist / 2.0, 0.0, 1.0)  # 0.0〜1.0
    reward += (safety_score - 0.5) * cfg.reward_safety_weight

    # EXP-10: センターライン報酬（左右非対称ペナルティの高度化）
    # LiDAR前方右側(0〜539), 後半左側(540〜1079)
    left_min  = np.min(scans[540:])
    right_min = np.min(scans[:540])
    total_width = left_min + right_min
    # センター度合い（0.0=真ん中, 1.0=どちらかの壁）
    center_ratio = abs(left_min - right_min) / (total_width + 1e-6)
    # 左右バランスが良いほどボーナスを与える（最大+0.3）
    center_bonus = (1.0 - center_ratio) * 0.3
    reward += center_bonus

    # EXP-11: 指数関数的な壁ペナルティ (Boundary Penalty)
    # 0.6m以内に近づいた場合のみ、急激にマイナスを増やす
    if wall_dist < 0.6:
        # e^(3.0 * 0.6) = e^1.8 ≒ 6.05。距離0で最大約 -6.0 の強烈な拒絶
        proximity_penalty = -np.exp(3.0 * (0.6 - wall_dist))
        reward += proximity_penalty

    # 5. 走行距離報酬（円形走行抑制）
    progress = np.sqrt((cur_x - prev_x) ** 2 + (cur_y - prev_y) ** 2)
    reward += progress * cfg.reward_progress_weight * progress_scale

    # 6. ステアリング安定性（条件付き）
    if front_dist > 5.0:
        reward += (1.0 - abs(action[0])) * 0.2
    reward += cfg.reward_survival

    return reward
