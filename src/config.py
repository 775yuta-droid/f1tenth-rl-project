import numpy as np
import os

import multiprocessing

from .profiles import PROFILES

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
TOTAL_TIMESTEPS = 10000000
LEARNING_RATE = 5e-5  # EXP-25 付近の標準的な学習率に戻す

# --- ネットワーク構造 ---
# 複雑な判断（加減速）をさせるため、階層を拡大 [64, 64] -> [128, 128] (EXP-07)
NET_ARCH = [128, 128]

# --- 観測空間の工夫 ---
LIDAR_BEAMS = 1440             # シミュレータの全周ビーム数 (360°分)
# 270°分を1080点とするため、360°では 1080 * 360 / 270 = 1440点 となる (0.25°刻み)
LIDAR_DOWNSAMPLE_FACTOR = 5   # EXP-39: 解像度を2倍に(10->5)
FRAME_STACK = 4                # スタックするフレーム数
FRAME_SKIP = 3                 # 間引き間隔 (40Hz時、Skip=3で0.25秒分をカバー)
N_ENVS = 1                      # テストのため 1 に削減 (元は 8)
INCLUDE_VEHICLE_STATE = True  # 速度とステアリング角を観測に含める
INCLUDE_LIDAR_RESIDUAL = False # ΔLiDAR は行動安定に寄与 (EXP-15: ノイズ排除のため無効化)

# --- 正規化設定 ---
NORMALIZE_OBSERVATIONS = True
LIDAR_MEAN = 2.188             # 0.25°刻み1440本仕様での平均
LIDAR_STD = 2.174              # 0.25°刻み1440本仕様での標準偏差
LIDAR_RESIDUAL_MEAN = 0.0
LIDAR_RESIDUAL_STD = 1.0
VEHICLE_STATE_MEAN = np.array([0.284, -0.005])  # [vel, steer]
VEHICLE_STATE_STD = np.array([0.187, 0.111])

# --- PPO 探索設定 ---
PPO_ENT_COEF = 0.03  # EXP-32: 0.03 -> 0.01 (Resume時の探索を抜い、EXP-25の知識を活かす)

# --- 物理設定（マシン性能） ---
CONTROL_HZ = 40            # 実機LiDARに合わせた制御周波数 (40Hz)
SIM_TIMESTEP = 1.0 / CONTROL_HZ
STEER_SENSITIVITY = 1.0    # EXP-35: 1.3 -> 1.0 に復帰 (元の感度に戻し、AIの運転感覚の狂いを解消)
MIN_SPEED = float(os.environ.get("MIN_SPEED", "0.3"))
MAX_SPEED = float(os.environ.get("MAX_SPEED", "2.5"))  # EXP-25: 2.5m/s

# --- マシン寸法 ---
CAR_LENGTH = 0.465
CAR_WIDTH = 0.19           # EXP-35: 0.23 -> 0.19 に復帰 (太さを元に戻し、物理的に狭いコースを曲がれるようにする)

# --- 報酬設計の設定 ---
REWARD_COLLISION = -100.0
REWARD_SURVIVAL  = 0.2     # EXP-25: 0.2
REWARD_FRONT_WEIGHT = 3.0   # 前方の空きスペースに対する報酬の重み
REWARD_SPEED_WEIGHT = 2.0   # EXP-26: 1.0 -> 2.0 (速度報酬の重みを2倍に)
REWARD_SAFETY_WEIGHT = 0.8  # 壁との安全距離スコア報酬
REWARD_DISTANCE_WEIGHT = 1.0   # 壁接近ペナルティ
REWARD_PROGRESS_WEIGHT = 4.0   # EXP-26: 2.0 -> 4.0 (走行距離報酬の重みを2倍にし、高速走破を奨励)

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

MAP_PATH  = os.environ.get("MAP_PATH",  "/workspace/my_maps/testmap-tamoku/map-tamoku")
MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/models")
LOG_DIR   = os.environ.get("LOG_DIR",   "/workspace/logs")

# --- 初期位置設定 [x, y, yaw] ---
# view_spawn.py で確認しながら調整してください
START_POSE = [-2.2, -3.5, 0.0]

# スタート位置のランダム化（Trueの場合、下記リストからランダムに選択）
START_POSE_RANDOMIZE = True
START_POSES = [
    [4.2, -0.2,  3.2],
    [-1.8, -0.1, 3.1],
    [7.5, -3.5, 0.0],
    [-2.2, -3.5,  0.0],
    # [4.5, 4.4,  2.0],  # EXP-25: さらに除外 (残りの衝突20%の主因)
    # [0.7, 5.0, -1.0],  # EXP-24: 除外
    #[5.0, 4.5, -2.5],
]

# モデル名に設定を反映させて管理しやすくする
MAP_NAME   = os.path.basename(MAP_PATH)
MODEL_NAME = f"ppo_f1_{MAP_NAME}_steps{TOTAL_TIMESTEPS}_arch{len(NET_ARCH)}"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
GIF_DIR    = os.path.join(PROJECT_ROOT, "gif")
GIF_PATH   = os.path.join(GIF_DIR, f"run_simulation_{MAP_NAME}_steps{TOTAL_TIMESTEPS}_arch{len(NET_ARCH)}.gif")

# 報酬計算ロジックは src/rewards.py に移動しました。
# from .rewards import calculate_reward