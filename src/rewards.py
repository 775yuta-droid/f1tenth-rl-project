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
    # EXP-32: Hokuyo URG 270° マスキング (Sim-to-Real ギャップ解消)
    # 実機のHokuyo LiDARは後方90°が不可視。後方のインデックスを除外する。
    # 1080点(360°) → 810点(270°): 後方 135点ずつを除外
    #   Index: 0=右後方(-135°), 405=正前方(0°), 810=左後方(+135°)
    # ================================================================
    _H_START, _H_END = 135, 945
    s = scans[_H_START:_H_END]  # 810点: -135°〜+135°

    # 1. 前方空間報酬 (±60° = 810点の中心405±180点)
    front_dist = np.min(s[225:585])
    reward = (front_dist / 30.0) * cfg.reward_front_weight

    # 2. 側面壁距離を個別に取得 (狭い直線・カーブ対策の核心)
    #    270°の場合: 右90° ≒ index 135付近, 左90° ≒ index 675付近
    right_side = np.min(s[90:270])    # 右方向 (-90°±45°帯)
    left_side  = np.min(s[540:720])   # 左方向 (+90°±45°帯)
    side_min   = min(right_side, left_side)

    # 3. 速度報酬 / コーナー前補正 (EXP-32: 4段階評価)
    speed_factor = current_speed / cfg.max_speed
    if front_dist < 2.0:           # 壁が目前: 強制ブレーキ
        reward -= speed_factor * cfg.reward_speed_weight * 3.0
        progress_scale = 0.05
    elif front_dist < 4.0:         # コーナー手前: 早めに減速開始
        reward += speed_factor * cfg.reward_speed_weight * 0.05
        progress_scale = 0.4
    elif front_dist < 7.0:         # 中間距離: 慎重に加速
        reward += speed_factor * cfg.reward_speed_weight * 0.6
        progress_scale = 0.8
    else:                          # 直線: 加速推奨
        reward += speed_factor * cfg.reward_speed_weight
        progress_scale = 1.0

    # 4. 安全距離スコア（全方位・連続関数で滑らかな勾配）
    wall_dist = np.min(s)
    safety_score = np.clip(wall_dist / 2.0, 0.0, 1.0)
    reward += (safety_score - 0.5) * cfg.reward_safety_weight

    # 5. センターライン維持【EXP-32 最優先: 二乗ペナルティ化】
    # 狭い直線でもドリフトは即死のため、中心を外すほど急激にマイナス
    total_width  = left_side + right_side
    center_ratio = abs(left_side - right_side) / (total_width + 1e-6)  # 0=中央, 1=壁
    center_penalty = -(center_ratio ** 2) * 3.0  # 二乗: わずかなズレも強く罰する
    reward += center_penalty

    # 6. 側面壁への超近距離ペナルティ【狭い直線対策: デッドライン0.35m】
    # マシン横幅0.19m + 安全マージン ≈ 0.35mを死線とする
    if side_min < 0.35:
        reward += -np.exp(6.0 * (0.35 - side_min))   # 係数3→6に増強

    # 7.「狭いカーブ」複合ペナルティ【EXP-32 核心: 前方詰まり × 側面接近の同時発生】
    # 最も危険な状況（前が閉じていて左右も余裕がない）に重ペナルティ
    in_narrow_curve = (front_dist < 4.5) and (side_min < 0.7)
    if in_narrow_curve:
        reward -= speed_factor * 4.0          # スピードを出しているほど強いペナルティ
        reward += (1.0 - abs(action[0])) * 0.8  # じわっとしたステアを強く評価

    # 8. 全方位近接ペナルティ (デッドライン 0.6→0.35m に厳格化)
    if wall_dist < 0.35:
        reward += -np.exp(4.0 * (0.35 - wall_dist))

    # 9. 走行距離報酬（進行を促進、円形走行抑制）
    progress = np.sqrt((cur_x - prev_x) ** 2 + (cur_y - prev_y) ** 2)
    reward += progress * cfg.reward_progress_weight * progress_scale

    # 10. ステアリング安定性（中速域でも適用: front_dist > 3m で有効）
    if front_dist > 3.0:
        reward += (1.0 - abs(action[0])) * 0.4   # 係数 0.3→0.4

    reward += cfg.reward_survival

    return reward
