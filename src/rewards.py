"""
報酬計算モジュール (map_1_0509_145516 最適化版 v2 / EXP-49)

マップ実測値:
  実寸: 7.85 x 7.90 m
  壁距離 p25=0.20m / p50=0.40m / p75=0.68m / max=2.51m
  LiDAR実効最大距離: ~7.9m（/8.0 で正規化）

v2 改善点:
  [Fix-R1] effective_front: open_side を廃止
             旧: max(front_dist, open_side * 0.8) → 斜め向きで回転ハッキング
             新: front_dist のみ → 真正面に空間があることを要求

  [Fix-R2] カーブステアリング報酬（新規）
             左右前方距離の非対称性を検出し、開いている側へのステアリングを報酬化
             「曲がりきれた」成功体験をRLに与える

  [Fix-R3] ステアリング安定性ボーナスの条件修正
             旧: front_dist > 0.5 → カーブ入口でも直進を促す
             新: front_dist > 1.0 かつ asymmetry < 0.15（直線時のみ）

  [Fix-R4] スピンペナルティ強化
             係数 0.5→2.0, 進捗閾値 0.02m→0.05m, スラック 0.3→0.2
"""

from dataclasses import dataclass
import numpy as np
from . import config


# ============================================================
# マップ固有定数 (map_1_0509_145516 実測値)
# ============================================================
MAP_LIDAR_EFFECTIVE_RANGE = 8.0   # m: マップ実質最大見通し距離
MAP_WALL_DIST_P50 = 0.40          # m: 壁距離中央値（safety_score の正規化基準）
MAP_WALL_DIST_P75 = 0.68          # m: 壁距離p75（「安全」の定義上限）
MAP_WALL_DIST_DANGER = 0.15       # m: 危険ゾーン境界

BRAKE_TIME_COEFF = 0.8   # 反応時間係数 [s]
BRAKE_MARGIN     = 0.5   # 余裕距離 [m]

# カーブ検出閾値
# asymmetry = |diag_left - diag_right| / (diag_left + diag_right)
# 0.0 = 完全直線, 1.0 = 完全カーブ
CURVE_ASYMMETRY_THRESHOLD = 0.20
STRAIGHT_ASYMMETRY_MAX    = 0.15  # この値以下なら「直線」とみなす


@dataclass
class RewardConfig:
    reward_collision: float       = -200.0
    reward_survival: float        = 0.3
    reward_front_weight: float    = 3.0
    reward_speed_weight: float    = 1.5
    reward_safety_weight: float   = 0.8
    reward_distance_weight: float = 1.0
    reward_progress_weight: float = 1.0
    reward_curve_weight: float    = 1.2   # [Fix-R2] カーブステアリング報酬の重み
    max_speed: float              = 2.5


def _load_default_config() -> RewardConfig:
    return RewardConfig(
        reward_collision=config.REWARD_COLLISION,
        reward_survival=config.REWARD_SURVIVAL,
        reward_front_weight=config.REWARD_FRONT_WEIGHT,
        reward_speed_weight=config.REWARD_SPEED_WEIGHT,
        reward_safety_weight=config.REWARD_SAFETY_WEIGHT,
        reward_distance_weight=config.REWARD_DISTANCE_WEIGHT,
        reward_progress_weight=config.REWARD_PROGRESS_WEIGHT,
        reward_curve_weight=config.REWARD_CURVE_WEIGHT,
        max_speed=config.MAX_SPEED,
    )


def calculate_reward(
    scans,
    action,
    done: bool,
    current_speed: float,
    prev_x: float = 0.0,
    prev_y: float = 0.0,
    cur_x:  float = 0.0,
    cur_y:  float = 0.0,
    reward_config: RewardConfig = None,
) -> float:
    cfg = reward_config if reward_config is not None else _load_default_config()

    if done:
        return cfg.reward_collision

    # ----------------------------------------------------------
    # Hokuyo 270° マスキング
    # s: 1080点, インデックス0=左端(-135°), 540=正面(0°), 1079=右端(+135°)
    # ----------------------------------------------------------
    _H_START, _H_END = 180, 1260
    s = scans[_H_START:_H_END]   # 1080点: -135°〜+135°

    # ----------------------------------------------------------
    # 前方・斜め距離の計算
    # s[380:700]: 前方±20° (正面±160/540*135°)
    # s[540:700]: 前方左 (0°〜+20°)
    # s[380:540]: 前方右 (-20°〜0°)
    # ----------------------------------------------------------
    front_dist = np.min(s[380:700])
    diag_left  = np.min(s[540:700])   # 前方左斜め
    diag_right = np.min(s[380:540])   # 前方右斜め

    # カーブ方向の非対称性 [0, 1]
    lr_sum    = diag_left + diag_right + 1e-6
    asymmetry = abs(diag_left - diag_right) / lr_sum

    # ----------------------------------------------------------
    # 1. 前方空間報酬 [Fix-R1: open_side 廃止]
    #
    # 旧: effective_front = max(front_dist, open_side * 0.8)
    #      斜めを向くだけで open_side が大きくなり高報酬
    #      → 回転し続けることで常に高報酬を得られる（報酬ハッキング）
    #
    # 新: effective_front = front_dist
    #      真正面に空間があることを要求
    #      → 回転しても前方は壁になるため報酬ハッキングを根絶
    # ----------------------------------------------------------
    effective_front = front_dist

    reward = (
        np.clip(effective_front, 0.0, MAP_LIDAR_EFFECTIVE_RANGE)
        / MAP_LIDAR_EFFECTIVE_RANGE
    ) * cfg.reward_front_weight

    # ----------------------------------------------------------
    # 2. 側面壁距離
    # ----------------------------------------------------------
    right_side = np.min(s[0:360])
    left_side  = np.min(s[720:1080])

    # ----------------------------------------------------------
    # 3. 速度報酬
    #
    # safe_brake_dist = v*0.8 + 0.5
    #   MIN_SPEED=0.3 → 0.74m, MAX_SPEED=2.5 → 2.50m
    # ----------------------------------------------------------
    speed_factor    = current_speed / cfg.max_speed
    safe_brake_dist = current_speed * BRAKE_TIME_COEFF + BRAKE_MARGIN

    if front_dist < safe_brake_dist:
        danger_ratio = 1.0 - (front_dist / safe_brake_dist)
        reward      -= speed_factor * cfg.reward_speed_weight * (2.0 + 3.0 * danger_ratio)
        progress_scale = 0.5
    else:
        reward        += speed_factor * cfg.reward_speed_weight
        progress_scale = 1.0

    # ----------------------------------------------------------
    # 4. 安全距離スコア（3段階ゾーン評価）
    # ----------------------------------------------------------
    wall_dist = np.min(s)

    if wall_dist < MAP_WALL_DIST_DANGER:
        safety_score = -1.0
    elif wall_dist < MAP_WALL_DIST_P50:
        t = (wall_dist - MAP_WALL_DIST_DANGER) / (MAP_WALL_DIST_P50 - MAP_WALL_DIST_DANGER)
        safety_score = -1.0 + t          # [-1.0, 0.0]
    elif wall_dist < MAP_WALL_DIST_P75:
        t = (wall_dist - MAP_WALL_DIST_P50) / (MAP_WALL_DIST_P75 - MAP_WALL_DIST_P50)
        safety_score = t                  # [0.0, 1.0]
    else:
        safety_score = 1.0

    reward += safety_score * cfg.reward_safety_weight

    # ----------------------------------------------------------
    # 5. センターライン維持
    #    ※ このマップでは front_dist < 5.0 がほぼ常時成立するため
    #      center_penalty は実質ゼロ。意図的にそのまま残す。
    # ----------------------------------------------------------
    total_width  = left_side + right_side
    center_ratio = abs(left_side - right_side) / (total_width + 1e-6)
    if front_dist < 5.0:
        center_penalty = 0.0
    else:
        center_penalty = -center_ratio * 3.0
    reward += center_penalty

    # ----------------------------------------------------------
    # 6. 走行距離報酬
    # ----------------------------------------------------------
    progress = np.sqrt((cur_x - prev_x) ** 2 + (cur_y - prev_y) ** 2)
    reward  += progress * cfg.reward_progress_weight * progress_scale

    # ----------------------------------------------------------
    # 7. スピンペナルティ [Fix-R4: 係数・閾値強化]
    #
    # 旧: threshold=0.02m, slack=0.3, coeff=0.5
    #      → open_side の高報酬を相殺できず回転ハッキングを許容
    #
    # 新: threshold=0.05m, slack=0.2, coeff=2.0
    #      → open_side廃止後も確実に回転を抑制する二重安全装置
    # ----------------------------------------------------------
    progress_norm = np.clip(progress / 0.05, 0.0, 1.0)   # 0.02m → 0.05m
    spin_excess   = max(0.0, (1.0 - progress_norm) - 0.2) # slack 0.3 → 0.2
    spin_penalty  = abs(action[0]) * spin_excess * 2.0    # coeff 0.5 → 2.0
    reward       -= spin_penalty

    # ----------------------------------------------------------
    # 8. カーブステアリング報酬 [Fix-R2: 新規追加]
    #
    # 左右前方距離の非対称性からカーブ方向を推定し、
    # 開いている方向にステアリングを切る行動を報酬化する。
    #
    # open_dir:
    #   +1.0 = 左が開いている（左コーナー）→ 正ステアリングが正解
    #   -1.0 = 右が開いている（右コーナー）→ 負ステアリングが正解
    #
    # steer_alignment = action[0] * open_dir → [-1, 1]
    #   正 = 正しい方向へ切っている
    #   負 = 間違った方向へ切っている
    # ----------------------------------------------------------
    if asymmetry > CURVE_ASYMMETRY_THRESHOLD:
        open_dir        = 1.0 if diag_left > diag_right else -1.0
        steer_alignment = float(action[0]) * open_dir   # [-1, 1]
        curve_reward    = steer_alignment * asymmetry * cfg.reward_curve_weight
        reward         += curve_reward

    # ----------------------------------------------------------
    # 9. ステアリング安定性 [Fix-R3: 直線判定条件を追加]
    #
    # 旧: if front_dist > 0.5 → カーブ入口でも直進ボーナスが働く
    #      → 曲がるべき場面でも直進し続けることを学習してしまう
    #
    # 新: if front_dist > 1.0 AND asymmetry < STRAIGHT_ASYMMETRY_MAX
    #      → コースが直線の時だけ直進ボーナスを与える
    # ----------------------------------------------------------
    if front_dist > 1.0 and asymmetry < STRAIGHT_ASYMMETRY_MAX:
        reward += (1.0 - abs(action[0])) * 0.2   # 係数 0.1 → 0.2

    # ----------------------------------------------------------
    # 10. 生存報酬
    # ----------------------------------------------------------
    reward += cfg.reward_survival

    return reward
