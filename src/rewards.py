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
from . import config


@dataclass
class RewardConfig:
    """報酬計算に必要なハイパーパラメータをまとめた設定クラス。"""
    reward_collision: float = -1000.0
    reward_survival: float = 0.2       # EXP-25: 0.05 -> 0.2 (安定完走の鍵)
    reward_front_weight: float = 3.0
    reward_speed_weight: float = 1.0
    reward_safety_weight: float = 0.8  # 旧 centrality_weight + distance_weight を統合
    reward_distance_weight: float = 1.0   # 互換性のため残存
    reward_progress_weight: float = 1.0
    max_speed: float = 3.0             # EXP-30/31設定に合わせる


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

    # ================================================================
    # Hokuyo URG 270° マスキング (Sim-to-Real 対応)
    # 実機のHokuyo LiDARは後方90°が不可視。後方のインデックスを除外する。
    # 1080点(360°) → 810点(270°): 後方 135点ずつを除外
    # ================================================================
    _H_START, _H_END = 135, 945
    s = scans[_H_START:_H_END]  # 810点: -135°〜+135°

    # ================================================================
    # LiDARインデックスの角度対応 (810点 / 270° = 3点/度)
    #   s[0]   = -135°,  s[405] = 0°(前方),  s[810] = +135°
    #   角度θ → インデックス: (θ + 135) * 3
    # ================================================================

    # 1. 前方空間報酬 (EXP-38: ±60° → ±40° に絞って精度を向上)
    #    -40°: (−40+135)*3=285,  +40°: (40+135)*3=525
    front_dist = np.min(s[285:525])
    reward = (front_dist / 30.0) * cfg.reward_front_weight

    # 2. 斜め前方 (±40°〜±80°) の左右個別検出 — カーブ入口シグナル (EXP-38)
    #    右斜め前: -80°〜-40°  → s[165:285]  (-80°=(−80+135)*3=165)
    #    左斜め前: +40°〜+80°  → s[525:645]  (+80°=(80+135)*3=645)
    diag_right = np.min(s[165:285])
    diag_left  = np.min(s[525:645])
    diag_total = diag_left + diag_right + 1e-6

    # 前方が開いているのに斜め左右に大きな差 = カーブ入口に差し掛かっている
    # 正しいカーブ方向に操舵していればボーナス、していなければペナルティ
    if front_dist > 3.0:
        diag_asymmetry = abs(diag_left - diag_right) / diag_total
        # curve_dir: +1=左カーブ(左diag短い), -1=右カーブ(右diag短い)
        curve_dir = np.sign(diag_left - diag_right)
        steer = action[0]
        steer_alignment = steer * curve_dir  # 正 = カーブ方向に操舵中

        if steer_alignment > 0:
            # カーブ方向に既に操舵 → ボーナス
            reward += diag_asymmetry * steer_alignment * 1.0
        else:
            # カーブ入口で直進 or 逆操舵 → ペナルティ
            reward -= diag_asymmetry * 1.5

    # 3. 側面壁距離の取得 (センターライン計算用)
    right_side = np.min(s[90:270])    # 右方向 (-90°±45°帯)
    left_side  = np.min(s[540:720])   # 左方向 (+90°±45°帯)

    # 4. 速度報酬 (EXP-25: 2.0m基準のシンプル補正 → EXP-39: 4.0mに拡大)
    speed_factor = current_speed / cfg.max_speed
    if front_dist < 4.0:           # 壁が目前: ブレーキ奨励 (EXP-39)
        reward -= speed_factor * cfg.reward_speed_weight * 2.0
        progress_scale = 0.5
    else:                          # 通常・直線: 加速推奨
        reward += speed_factor * cfg.reward_speed_weight
        progress_scale = 1.0

    # 5. 安全距離スコア
    wall_dist = np.min(s)
    safety_score = np.clip(wall_dist / 2.0, 0.0, 1.0)
    reward += (safety_score - 0.5) * cfg.reward_safety_weight

    # 6. センターライン維持 (EXP-25: 二乗なしの比率ペナルティ)
    total_width  = left_side + right_side
    center_ratio = abs(left_side - right_side) / (total_width + 1e-6)
    center_penalty = -center_ratio * 4.0  # EXP-25: シンプルなペナルティ
    reward += center_penalty

    # 7. 走行距離報酬
    progress = np.sqrt((cur_x - prev_x) ** 2 + (cur_y - prev_y) ** 2)
    reward += progress * cfg.reward_progress_weight * progress_scale

    # 8. ステアリング安定性 (EXP-38: カーブでの積極的操舵を妨げないよう軽減)
    reward += (1.0 - abs(action[0])) * 0.1  # 0.3 -> 0.1

    # 9. 生存報酬
    reward += cfg.reward_survival

    return reward
