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
    reward_line_weight: float     = 0.5   # 先生提案: レーシングライン誤差ペナルティ重み
    reward_smooth_weight: float   = 0.1   # 先生提案: 操作量の急変ペナルティ
    yaw_rate_penalty_weight: float = 1.5  # [Fix-R4] 角速度ペナルティの重み
    max_speed: float              = 2.5


# グローバルで1回だけ生成して使い回す
_DEFAULT_REWARD_CFG = None

def _get_cached_config():
    global _DEFAULT_REWARD_CFG
    if _DEFAULT_REWARD_CFG is None:
        _DEFAULT_REWARD_CFG = RewardConfig(
            reward_collision=config.REWARD_COLLISION,
            reward_survival=config.REWARD_SURVIVAL,
            reward_front_weight=config.REWARD_FRONT_WEIGHT,
            reward_speed_weight=config.REWARD_SPEED_WEIGHT,
            reward_safety_weight=config.REWARD_SAFETY_WEIGHT,
            reward_distance_weight=config.REWARD_DISTANCE_WEIGHT,
            reward_progress_weight=config.REWARD_PROGRESS_WEIGHT,
            reward_curve_weight=config.REWARD_CURVE_WEIGHT,
            reward_line_weight=config.REWARD_LINE_WEIGHT,
            reward_smooth_weight=config.REWARD_SMOOTH_WEIGHT,
            yaw_rate_penalty_weight=config.YAW_RATE_PENALTY_WEIGHT,
            max_speed=config.MAX_SPEED,
        )
    return _DEFAULT_REWARD_CFG



def calculate_reward(
    scans,
    action,
    done: bool,
    current_speed: float,
    prev_x: float = 0.0,
    prev_y: float = 0.0,
    cur_x:  float = 0.0,
    cur_y:  float = 0.0,
    cte_norm: float = 0.0,    # 先生提案: レーシングライン横誤差 (正規化済み [-1,1])
    curvature: float = 0.0,   # 先生提案: 前方曲率
    prev_action: np.ndarray = None,  # 滑らかさ計算用
    heading: float = 0.0,     # [Fix-R4] 現在の方位 (yaw [rad])
    yaw_rate: float = 0.0,    # [Fix-R4] 角速度 ([rad/step])
    reward_config: RewardConfig = None,
) -> float:
    cfg = reward_config if reward_config is not None else _get_cached_config()

    if done:
        return cfg.reward_collision

    # ----------------------------------------------------------
    # Hokuyo 270° マスキング
    # s: 1080点, インデックス0=左端(-135°), 540=正面(0°), 1079=右端(+135°)
    # すでに f1_env.py で 1080点にスライス済み。
    # ----------------------------------------------------------
    s = scans
    if len(s) > 1080:
        s = scans[180:1260] # 予備のスライス
    
    # 前方・斜め距離の計算 (一括スライスで高速化)
    # front_slice: 前方±20° (正面540に対して 380〜700)
    front_slice = s[380:700]
    front_dist  = np.min(front_slice)
    diag_right  = np.min(front_slice[:160]) # 右斜め (-20°〜0°)
    diag_left   = np.min(front_slice[160:]) # 左斜め (0°〜+20°)

    # カーブ方向の非対称性 [0, 1]
    lr_sum    = diag_left + diag_right + 1e-6
    asymmetry = abs(diag_left - diag_right) / lr_sum

    speed_factor    = current_speed / cfg.max_speed

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
    ) * cfg.reward_front_weight * speed_factor # [Fix-R1] 速度に比例させ、停止中の報酬をカット

    # ----------------------------------------------------------
    # 2. 側面壁距離（センターライン削除後は wall_dist のみ安全スコアで使用）
    # ----------------------------------------------------------
    # right_side / left_side: センターライン報酬削除に伴い不要化したため除去

    # ----------------------------------------------------------
    # 3. 速度報酬 [先生提案: r_speed = 現在速度 × コース曲率に応じた係数]
    #
    # 曲率が大きい（急カーブ）ほど、高い速度を出すことへの報酬を減らす（またはペナルティ化）。
    # これによりコーナー手前での自動的な減速を促す。
    # ----------------------------------------------------------
    
    # 曲率に基づくペナルティ係数 (曲率 0.0 で 1.0, 曲率 5.0 で 0.0 程度)
    curv_penalty_scale = max(0.0, 1.0 - abs(curvature) * 0.2)
    
    safe_brake_dist = current_speed * BRAKE_TIME_COEFF + BRAKE_MARGIN

    if front_dist < safe_brake_dist:
        danger_ratio = 1.0 - (front_dist / safe_brake_dist)
        reward      -= speed_factor * cfg.reward_speed_weight * (2.0 + 3.0 * danger_ratio)
        progress_scale = 0.5
    else:
        # 先生の式を反映: 曲率が大きいほど速度報酬が減衰する
        reward        += (speed_factor * curv_penalty_scale) * cfg.reward_speed_weight
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
    # 5. センターライン維持 → 削除済み（先生指摘: 実質ゼロで混乱の原因）
    #    旧: front_dist < 5.0 がほぼ常時成立するため center_penalty = 0.0 だった
    #    → コードを残すことでデバッグ時に誤読するリスクがあるため削除
    # ----------------------------------------------------------
    # 6. 走行距離報酬 [Fix-R4: 前進成分のみを抽出]
    # ----------------------------------------------------------
    dx = cur_x - prev_x
    dy = cur_y - prev_y
    # 現在の向きに対する射影成分（前進距離）を計算
    forward_progress = dx * np.cos(heading) + dy * np.sin(heading)
    forward_progress = max(0.0, forward_progress) # 後退は報酬なし
    
    reward += forward_progress * cfg.reward_progress_weight * progress_scale

    # ----------------------------------------------------------
    # 6b. レーシングライン誤差ペナルティ [先生提案: r_line]
    #
    # cte_norm: [-1, 1] の横方向誤差。外れるほど大きなペナルティ。
    # CSVが生成されていない場合は cte_norm=0.0 でスキップされる。
    # ----------------------------------------------------------
    r_line = -abs(cte_norm) * cfg.reward_line_weight
    reward += r_line

    # ----------------------------------------------------------
    # 7. 角速度ペナルティ (スピン防止) [Fix-R4]
    # ----------------------------------------------------------
    # 1ステップで 180度(pi rad) 以上回転している場合は異常とみなす
    yaw_rate_norm = abs(yaw_rate) / np.pi
    reward -= yaw_rate_norm * cfg.yaw_rate_penalty_weight

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
    # [Fix-R4] 前進が極端に少ない場合はカーブ報酬を与えない（その場旋回対策）
    if asymmetry > CURVE_ASYMMETRY_THRESHOLD and forward_progress > 0.02:
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
    # 11. 操作の滑らかさ [先生提案: r_smooth]
    # ----------------------------------------------------------
    if prev_action is not None:
        # 前ステップのアクションとの差分（ステアリング、速度）
        action_diff = np.abs(action - prev_action)
        # 急激な変化ほど大きなペナルティ
        reward -= np.sum(action_diff) * cfg.reward_smooth_weight

    return reward
