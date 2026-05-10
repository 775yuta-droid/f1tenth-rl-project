"""
報酬計算モジュール (map_1_0509_145516 最適化版)

マップ実測値:
  実寸: 7.85 x 7.90 m
  壁距離 p25=0.20m / p50=0.40m / p75=0.68m / max=2.51m
  LiDAR実効最大距離: ~7.9m（30mではなく8mで正規化）

修正点:
  [Fix-1] safe_brake_dist: マップスケールに合わせて定数を再計算
  [Fix-2] front_dist正規化: /30.0 -> /MAP_LIDAR_EFFECTIVE_RANGE(8.0)
  [Fix-3] safety_score: 正規化基準を p50=0.40m ベースに変更
"""

from dataclasses import dataclass
import numpy as np
from . import config


# ============================================================
# マップ固有定数 (map_1_0509_145516 実測値)
# ============================================================
MAP_LIDAR_EFFECTIVE_RANGE = 8.0   # m: マップ実質最大見通し距離（/30.0 → /8.0）
MAP_WALL_DIST_P50 = 0.40          # m: 壁距離中央値（safety_score の正規化基準）
MAP_WALL_DIST_P75 = 0.68          # m: 壁距離p75（「安全」の定義上限）
MAP_WALL_DIST_DANGER = 0.15       # m: 危険ゾーン境界（実測危険帯）

# safe_brake_dist の定数項を縮小
# 旧: v*1.5 + 2.0 → MIN_SPEED=0.3でも 2.45m（マップ幅を超える）
# 新: v*0.8 + 0.5 → MIN_SPEED=0.3で 0.74m, MAX_SPEED=2.5で 2.50m（マップ最大距離と一致）
BRAKE_TIME_COEFF = 0.8   # 反応時間係数 [s]
BRAKE_MARGIN     = 0.5   # 余裕距離 [m]


@dataclass
class RewardConfig:
    reward_collision: float       = -200.0
    reward_survival: float        = 0.3
    reward_front_weight: float    = 3.0
    reward_speed_weight: float    = 1.5
    reward_safety_weight: float   = 0.8
    reward_distance_weight: float = 1.0
    reward_progress_weight: float = 1.0
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
    # Hokuyo 270° マスキング (変更なし)
    # ----------------------------------------------------------
    _H_START, _H_END = 180, 1260
    s = scans[_H_START:_H_END]   # 1080点: -135°〜+135°

    # ----------------------------------------------------------
    # 1. 前方空間報酬 [Fix-2]
    #
    # 旧: effective_front / 30.0
    # 新: effective_front / MAP_LIDAR_EFFECTIVE_RANGE (8.0)
    #
    # 理由: このマップの最大見通しは ~7.9m。
    #       /30.0 では最大報酬が 26% しか発揮されず
    #       前進インセンティブが著しく弱くなっていた。
    # ----------------------------------------------------------
    front_dist = np.min(s[380:700])
    diag_left  = np.min(s[540:700])
    diag_right = np.min(s[380:540])
    open_side  = max(diag_left, diag_right)
    effective_front = max(front_dist, open_side * 0.8)

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
    # 3. 速度報酬 [Fix-1]
    #
    # 旧: safe_brake_dist = v*1.5 + 2.0
    #       → MIN_SPEED=0.3で 2.45m。マップ最大2.51m と拮抗し
    #          全速度域で danger_ratio>0 → progress_scale=0.5 固定
    #
    # 新: safe_brake_dist = v*BRAKE_TIME_COEFF + BRAKE_MARGIN
    #       = v*0.8 + 0.5
    #       → MIN_SPEED=0.3で 0.74m, MAX_SPEED=2.5で 2.50m
    #          低速の安全走行では speed_reward が正に働くようになる
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
    # 4. 安全距離スコア [Fix-3]
    #
    # 旧: safety_score = clip(wall_dist / 2.0, 0, 1)
    #       → p50=0.40m で score=0.20 → (0.20-0.5)*0.8 = -0.24
    #          普通に走っても常時マイナス
    #
    # 新: 3段階ゾーン評価（マップ実測値を直接使用）
    #   d < DANGER(0.15m) → -1.0（強ペナルティ）
    #   DANGER〜p50(0.40m) → 線形補間 -1.0〜0.0
    #   p50〜p75(0.68m)   → 線形補間  0.0〜+1.0
    #   p75以上           → +1.0（最大ボーナス）
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
    # 5. センターライン維持（変更なし）
    #    ※ このマップでは front_dist < 5.0 がほぼ常時成立するため
    #      center_penalty は実質ゼロ。意図的にそのまま残す。
    #      （狭小マップではセンターより「曲がれるライン」優先が正解）
    # ----------------------------------------------------------
    total_width  = left_side + right_side
    center_ratio = abs(left_side - right_side) / (total_width + 1e-6)
    if front_dist < 5.0:
        center_penalty = 0.0
    else:
        center_penalty = -center_ratio * 3.0
    reward += center_penalty

    # ----------------------------------------------------------
    # 6. 走行距離報酬（変更なし）
    # ----------------------------------------------------------
    progress = np.sqrt((cur_x - prev_x) ** 2 + (cur_y - prev_y) ** 2)
    reward  += progress * cfg.reward_progress_weight * progress_scale

    # ----------------------------------------------------------
    # 7. 回転ペナルティ（変更なし）
    # ----------------------------------------------------------
    progress_norm = np.clip(progress / 0.02, 0.0, 1.0)
    spin_excess   = max(0.0, (1.0 - progress_norm) - 0.3)
    spin_penalty  = abs(action[0]) * spin_excess * 0.5
    reward       -= spin_penalty

    # ----------------------------------------------------------
    # 8. ステアリング安定性（変更なし）
    # ----------------------------------------------------------
    if front_dist > 0.5:
        reward += (1.0 - abs(action[0])) * 0.1

    # ----------------------------------------------------------
    # 9. 生存報酬（変更なし）
    # ----------------------------------------------------------
    reward += cfg.reward_survival

    return reward
