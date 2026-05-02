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
    """報酬計算に必要なハイパーパラメータをまとめた設定クラス。

    デフォルト値は config.py の REWARD_* 定数と完全に一致させる。
    不一致があると単体テスト・デバッグ時に意図外の報酬計算が行われるリスクがある。
    """
    reward_collision: float = -100.0       # config.REWARD_COLLISION: -1000 は知見1(局所解の罠)と矛盾するため-100渡し
    reward_survival: float = 0.2           # config.REWARD_SURVIVAL
    reward_front_weight: float = 3.0       # config.REWARD_FRONT_WEIGHT
    reward_speed_weight: float = 1.5       # config.REWARD_SPEED_WEIGHT: Phase1安全優先値(3.0→完走確認後に引き上げ)
    reward_safety_weight: float = 0.8       # config.REWARD_SAFETY_WEIGHT: EXP-08で1.5は失敗確認済み
    reward_distance_weight: float = 1.0    # 互換性のため残存
    reward_progress_weight: float = 1.0    # config.REWARD_PROGRESS_WEIGHT
    max_speed: float = 2.5                 # config.MAX_SPEED: EXP-25知見に従い 2.5m/s


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
        scans: LiDARの距離データ (1440点)
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
    # 1440点(360°) → 1080点(270°): 後方 180点ずつを除外 (1440 - 1080 = 360)
    # ================================================================
    _H_START, _H_END = 180, 1260
    s = scans[_H_START:_H_END]  # 1080点: -135°〜+135° (1点 = 0.25°)

    # ================================================================
    # LiDARインデックスの角度対応 (1080点 / 270° = 4点/度)
    #   s[0]   = -135°,  s[540] = 0°(前方),  s[1080] = +135°
    #   角度θ → インデックス: (θ + 135) * 4
    # ================================================================

    # 1. 前方空間報酬 (±40°)
    #    -40°: (−40+135)*4=380,  +40°: (40+135)*4=700
    front_dist = np.min(s[380:700])
    reward = (front_dist / 30.0) * cfg.reward_front_weight

    # 2. Follow the Gap 報酬 (EXP-46: 最も開いている方向へハンドルを切る動機付け)
    gap_idx = np.argmax(s) # 1080点の中で最も遠い方向
    gap_angle_deg = (gap_idx / 1080.0 * 270.0) - 135.0
    gap_angle_norm = gap_angle_deg / 135.0 # [-1, 1]
    
    steer = action[0] # 正=左, 負=右
    # gap_angle_norm > 0 (左に隙間) かつ steer > 0 (左操舵) なら正
    gap_alignment = steer * gap_angle_norm

    if abs(gap_angle_deg) > 15.0: # 真正面以外に隙間がある場合
        # 隙間に向かっていればボーナス (最大 0.5)
        reward += np.clip(gap_alignment, 0.0, 0.5)
    
    # 3. 側面壁距離の取得 (センターライン計算用)
    right_side = np.min(s[0:360])
    left_side  = np.min(s[720:1080])

    # 4. 速度報酬 (動的ブレーキ距離の導入)
    speed_factor = current_speed / cfg.max_speed
    
    # 速度が速いほど、遠くからブレーキをかける必要がある (1.5秒先の到達距離 + 余裕2.0m)
    safe_brake_dist = current_speed * 1.5 + 2.0 

    if front_dist < safe_brake_dist:
        # 壁に近づくほど、速度を出していることへのペナルティを強める（滑らかな連続関数）
        danger_ratio = 1.0 - (front_dist / safe_brake_dist)
        reward -= speed_factor * cfg.reward_speed_weight * (2.0 + 3.0 * danger_ratio)
        progress_scale = 0.5
    else:
        # 安全圏なら加速を推奨
        reward += speed_factor * cfg.reward_speed_weight
        progress_scale = 1.0

    # 5. 安全距離スコア
    wall_dist = np.min(s)
    safety_score = np.clip(wall_dist / 2.0, 0.0, 1.0)
    reward += (safety_score - 0.5) * cfg.reward_safety_weight

    # 6. センターライン維持 (EXP-46: カーブ接近時は中央維持を緩和し、デッドロックを回避)
    total_width  = left_side + right_side
    center_ratio = abs(left_side - right_side) / (total_width + 1e-6)
    if front_dist < 5.0:
        # カーブ進入時はセンターを外れることを許容
        center_penalty = 0.0
    else:
        center_penalty = -center_ratio * 3.0
    reward += center_penalty

    # 7. 走行距離報酬
    progress = np.sqrt((cur_x - prev_x) ** 2 + (cur_y - prev_y) ** 2)
    reward += progress * cfg.reward_progress_weight * progress_scale

    # 8. ステアリング安定性 (EXP-38: カーブでの積極的操舵を妨げないよう軽減)
    reward += (1.0 - abs(action[0])) * 0.1  # 0.3 -> 0.1

    # 9. 生存報酬 (EXP-45: config.REWARD_SURVIVALを0.5に引き上げ。「生きている間は必ず正の基礎報酬」を保証)
    reward += cfg.reward_survival

    return reward
