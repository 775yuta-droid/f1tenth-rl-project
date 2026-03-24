import numpy as np
import os

import multiprocessing

from src.profiles import PROFILES

# --- デバイス設定 ---
# 互換性重視のため CPU を指定
DEVICE = "cpu"  # "cpu", "cuda", "auto" から選択可能

# --- 学習環境プロファイル ---
# 環境変数 TRAINING_PROFILE で使用する設定セットを切り替えます。
#   laptop  : RTX 3050 Laptop 向け（TORCH_NUM_THREADS=2）
#   desktop : RTX 5080 Desktop 向け（TORCH_NUM_THREADS=4）
#   auto    : CPUコア数から自動判定（デフォルト）
#
# 例: TRAINING_PROFILE=laptop python3 scripts/train.py
_profile_name = os.environ.get("TRAINING_PROFILE", "auto")
_profile = PROFILES.get(_profile_name, PROFILES["auto"])

# --- CPU スレッド最適化 ---
# F1Tenthシミュレーターはシングルスレッド動作のため、
# PyTorchのスレッドが多すぎるとオーバーヘッドで逆に遅くなる。
# TRAINING_PROFILE で laptop/desktop を指定すると最適値が自動設定される。
# 手動で上書きしたい場合: TORCH_NUM_THREADS=N python3 scripts/train.py
_cpu_count = multiprocessing.cpu_count()
_profile_threads = _profile["torch_num_threads"]
if _profile_threads is None:
    # auto: CPUコア数から保守的に2スレッドに制限
    _profile_threads = 2 if _cpu_count >= 2 else 1
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", _profile_threads))


# --- 学習ハイパーパラメータ ---
# ステアリング+速度の2次元学習は時間がかかるため、300,000〜500,000を推奨
TOTAL_TIMESTEPS = 5000000
LEARNING_RATE = 5e-5  # EXP-11: 高速域での微調整のため慎重な学習率に変更 (1e-4 -> 5e-5)

# --- ネットワーク構造 ---
# 複雑な判断（加減速）をさせるため、階層を拡大 [64, 64] -> [128, 128] (EXP-07)
NET_ARCH = [128, 128]

# --- 観測空間の工夫 ---
LIDAR_DOWNSAMPLE_FACTOR = 10   # 1080 -> 540次元（残差処理のため高解像度を維持）
INCLUDE_VEHICLE_STATE = True  # 速度とステアリング角を観測に含める
INCLUDE_LIDAR_RESIDUAL = False # ΔLiDAR は行動安定に寄与 (EXP-06〜) - Trueにするなら単独実験で検証すること

# --- 正規化設定 ---
NORMALIZE_OBSERVATIONS = True
# Calibrated statistics based on 10000 random steps
LIDAR_MEAN = 4.869
LIDAR_STD = 3.577
LIDAR_RESIDUAL_MEAN = -0.008
LIDAR_RESIDUAL_STD = 0.084
VEHICLE_STATE_MEAN = np.array([0.574, -0.010])  # [vel, steer]
VEHICLE_STATE_STD = np.array([0.096, 0.122])

# --- PPO 探索設定 ---
PPO_ENT_COEF = 0.01  # エントロピー係数（収束優先・局所解は報酬設計で対処）

# --- 物理設定（マシン性能） ---
STEER_SENSITIVITY = 1.0   # ステアリングの反応速度
MIN_SPEED = 0.3            # EXP-13: コーナーでのブレーキを許可するため 1.0 -> 0.3 へ引き下げ
MAX_SPEED = 2.5            # 最高速度（3.0→コーナーで安全な速度に下げ）

# --- マシン寸法 ---
CAR_LENGTH = 0.465
CAR_WIDTH = 0.19

# --- 報酬設計の設定 ---
REWARD_COLLISION = -200.0  # ペナルティを緩和
REWARD_SURVIVAL  = 0.1     # 生存報酬を少し増やして補完
REWARD_FRONT_WEIGHT = 3.0   # 前方の空きスペースに対する報酬の重み
REWARD_SPEED_WEIGHT = 2.0   # 速度に対する報酬の重み (EXP-14: 1.0 -> 2.0)
REWARD_SAFETY_WEIGHT = 0.8  # 壁との安全距離スコア報酬（EXP-10で0.8に戻し、中央ボーナスを主軸に）
REWARD_DISTANCE_WEIGHT = 1.0   # 壁接近ペナルティ（safety_weightと役割統合済み・互換用）
REWARD_PROGRESS_WEIGHT = 2.0   # 走行距離報酬（EXP-10で1.0→2.0へ倍増、完走を最優先）

# --- パス設定 ---
# 環境変数で上書き可能。未設定の場合は Docker 内デフォルト値を使用。
#   MAP_PATH: 使用するマップ（拡張子なし）
#   MODEL_DIR: モデルの保存先
#   LOG_DIR:   TensorBoard ログの保存先
#
# 利用可能なマップ:
#   levine        -- 定番の廊下マップ
#   skirk         -- テストコース風
#   berlin        -- 市街地コース風
#   vegas         -- ラスベガス風
#   stata_basement -- 複雑な地下通路マップ
#   my_map        -- 独自の倉庫マップ（デフォルト）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAP_PATH  = os.environ.get("MAP_PATH",  "/workspace/my_maps/my_map")
MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/models")
LOG_DIR   = os.environ.get("LOG_DIR",   "/workspace/logs")

# --- 初期位置設定 [x, y, yaw] ---
# view_spawn.py で確認しながら調整してください
START_POSE = [2.5, 4.0, 0.0]

# スタート位置のランダム化（Trueの場合、下記リストからランダムに選択）
START_POSE_RANDOMIZE = True
START_POSES = [
    [1.5, 3.5,  0.5],
   # [3.0, 5.0,  2.5],
    [3.0, 5.0,  2.5],
    [4.5, 4.4,  2.0],
    [0.7, 5.0, -1.0],
    #[5.0, 4.5, -2.5],
]

# モデル名に設定を反映させて管理しやすくする
MAP_NAME   = os.path.basename(MAP_PATH)
MODEL_NAME = f"ppo_f1_{MAP_NAME}_steps{TOTAL_TIMESTEPS}_arch{len(NET_ARCH)}"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
GIF_DIR    = os.path.join(PROJECT_ROOT, "gif")
GIF_PATH   = os.path.join(GIF_DIR, f"run_simulation_{MAP_NAME}_steps{TOTAL_TIMESTEPS}_arch{len(NET_ARCH)}.gif")

# 報酬計算ロジックは src/rewards.py に移動しました。
# from src.rewards import calculate_reward