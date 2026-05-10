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
    reward_collision: float = -200.0       # config.REWARD_COLLISION: -1000 は知見1(局所解の罠)と矛盾するため-100渡し
    reward_survival: float = 0.3           # config.REWARD_SURVIVAL
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

    # 1. 前方・斜め前方空間報酬 (EXP-48改善)
    #    前方 ±40°: s[380:700]
    #    左斜め前  0°〜+40°: s[540:700]
    #    右斜め前 -40°〜0°: s[380:540]
    #
    #    tamoku マップの問題:
    #    カーブ入口では「正面は遠い・横が詰まる」という変化で現れる。
    #    front_dist だけでは検出が遅れ、入口で突然衝突する。
    #    斜め前方の広い側を 0.8 倍のウェイトで加味することで
    #    カーブ入口の「通れる隙間」を事前に報酬に反映する。
    front_dist = np.min(s[380:700])
    diag_left  = np.min(s[540:700])   # 左斜め前: 0°〜+40°
    diag_right = np.min(s[380:540])   # 右斜め前: -40°〜0°
    open_side  = max(diag_left, diag_right)
    # カーブ入口で片側が開いていれば、その空間を報酬として加味
    effective_front = max(front_dist, open_side * 0.8)
    reward = (effective_front / 30.0) * cfg.reward_front_weight

    # 2. 側面壁距離の取得 (センターライン計算用)
    #    右方向: -135°〜-45°  → s[0:360]
    #    左方向: +45°〜+135°  → s[720:1080]
    right_side = np.min(s[0:360])
    left_side  = np.min(s[720:1080])
    # ※ EXP-47: Follow the Gap 報酬を削除。
    # np.argmax(s)を使ったボーナスは「広い直線で回転し続ける」報酬ハッキングを誘発した(知見22)。

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

    # 8. 回転ペナルティ (EXP-48改善: 低速カーブ進入への誤爆を防止)
    # 「大きなステアリング入力のわりに前進距離が少ない」状態を検出してペナルティ
    #
    # EXP-47の問題:
    #   閾値 0.05m/step (≈2m/s) に対して、0.3m/s走行では
    #   progress ≈ 0.03m となり progress_norm ≈ 0.6 が常時発生。
    #   カーブで正常にハンドルを切るたびに毎ステップペナルティが課されていた。
    #
    # EXP-48改善:
    #   閾値を 0.02m/step (≈0.8m/s) に下げ、低速でも前進していればペナルティなし。
    #   さらにバッファ (0.3) を設け、「ほぼ停止+大ステア」のみを対象とする。
    #   これにより「低速カーブ走行」と「その場スピン」を明確に区別する。
    progress_norm = np.clip(progress / 0.02, 0.0, 1.0)  # 0.05 -> 0.02
    spin_excess = max(0.0, (1.0 - progress_norm) - 0.3)  # 0.3未満は無罰
    spin_penalty = abs(action[0]) * spin_excess * 0.5
    reward -= spin_penalty

    # 9. ステアリング安定性 (EXP-48改善: カーブ手前では直進バイアスをオフ)
    # 前方 > 5m の直線区間のみ「真っ直ぐ走る」を優遇する。
    # 前方 < 5m（カーブ入口）ではこの報酬を停止し、
    # 大きくハンドルを切る行動を妨げないようにする。
    if front_dist > 0.5:
        reward += (1.0 - abs(action[0])) * 0.1

    # 10. 生存報酬 (EXP-47: 0.2を維持。回転ペナルティ導入で「回転しながら生きる」期待報酬はマイナスに)
    reward += cfg.reward_survival

    return reward
