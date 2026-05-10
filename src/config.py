import numpy as np
import os

import multiprocessing

from .profiles import PROFILES

# --- デバイス設定 ---
# 互換性重視のため CPU を指定
DEVICE = "cuda"  # "cpu", "cuda", "auto" から選択可能

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
LEARNING_RATE = 5e-5  # EXP-25 付近の標準的な学習率に戻す
PPO_BATCH_SIZE = 512  # GPU効率化のため 64 -> 512
PPO_N_STEPS = 2048    # 1環境あたり収集するデータ

# --- ネットワーク構造 ---
# EXP-47: [128, 128] -> [256, 256] (features_dim=512に対するボトルネック解消。情報の取りこぼしを防ぐ)
NET_ARCH = [256, 256]

# --- 観測空間の工夫 ---
LIDAR_BEAMS = 1440             # シミュレータの全周ビーム数 (360°分)
# 270°分を1080点とするため、360°では 1080 * 360 / 270 = 1440点 となる (0.25°刻み)
LIDAR_DOWNSAMPLE_FACTOR = 5   # EXP-39: 解像度を2倍に(10->5)
FRAME_STACK = 4                # スタックするフレーム数
FRAME_SKIP = 4                 # EXP-44: 1 -> 4 (視野を 0.1s -> 0.4s に拡大し、CNNが動きを捉えやすくする)
N_ENVS = 8                     # テストのため 1 に削減 (元は 8)
INCLUDE_VEHICLE_STATE = True  # 速度とステアリング角を観測に含める
INCLUDE_LIDAR_RESIDUAL = False # ΔLiDAR は行動安定に寄与 (EXP-15: ノイズ排除のため無効化)
INCLUDE_EXTRA_FEATURES = False # 切り分けテスト: 観測をシンプル化 (後で再有効化)

# --- 正規化設定 ---
# development-plan.md 推奨: Z-score ではなく 0-1 反転方式を採用
#   lidar_norm = 1.0 - clip(lidar, 0, MAX_RANGE) / MAX_RANGE
#   → 近い壁 = 1.0, 遠い空間 = 0.0  (NaN/inf を安全にクリップ後に適用)
NORMALIZE_OBSERVATIONS = True
LIDAR_MAX_RANGE = 30.0         # クリッピング上限 (m)

# --- CNN ポリシー設定 ---
# True: Conv1DLidarExtractor + MlpPolicy, False: 従来の MlpPolicy (MLP のみ)
USE_CNN_POLICY = True

# --- PPO 探索設定 ---
PPO_ENT_COEF = 0.02  # EXP-47: 0.03 -> 0.01 (回転ハッキング収束後のstd爆発を防止。EXP-45で安定が確認された値)

# --- 物理設定（マシン性能） ---
CONTROL_HZ = 40            # 実機LiDARに合わせた制御周波数 (40Hz)
ACTION_REPEAT = 4          # EXP-44: 1 -> 4 (10Hz制御に落とし、低速時の挙動を安定化)
SIM_TIMESTEP = 1.0 / CONTROL_HZ
STEER_SENSITIVITY = 1.0    # EXP-35: 1.3 -> 1.0 に復帰 (元の感度に戻し、AIの運転感覚の狂いを解消)
MIN_SPEED = float(os.environ.get("MIN_SPEED", "0.3"))   # EXP-25知見: 0.3m/sでコーナーブレーキ許可
MAX_SPEED = float(os.environ.get("MAX_SPEED", "2.5"))   # EXP-25知見: 2.5m/sから段階的引き上げ。EXP-26〜29で高速 Fresh学習は全滅

# --- マシン寸法 ---
CAR_LENGTH = 0.465
CAR_WIDTH = 0.19           # EXP-35: 0.23 -> 0.19 に復帰 (太さを元に戻し、物理的に狭いコースを曲がれるようにする)

# --- 報酬設計の設定 ---
REWARD_COLLISION = -200.0
REWARD_SURVIVAL  = 0.3     # EXP-47: 0.5 -> 0.2 (「回転しながら生き続ける」報酬ハッキングを抑制。知見22)
REWARD_FRONT_WEIGHT = 3.0   # EXP-47: 3.0 -> 2.0 (広い空間を向くだけで得点が高すぎ、回転の動機になっていた)
REWARD_SPEED_WEIGHT = 1.5   # EXP-41 Phase1: 3.0→1.5 (安全優先。完走確認後に2.0→3.0へ引き上げ)
REWARD_SAFETY_WEIGHT = 0.8  # EXP-08知見: 1.5は「安全を求めすぎて逃げ場を失い逐に衝突」を引き起こすため 0.8 に戻す
REWARD_DISTANCE_WEIGHT = 1.0   # 壁接近ペナルティ
REWARD_PROGRESS_WEIGHT = 1.0   # EXP-45: 4.0 -> 1.0 (ACTION_REPEAT=4と積み合わさると実質16倍の爆発を招いた。生存優先に戻す)

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

MAP_PATH  = os.environ.get("MAP_PATH",  "/workspace/my_maps/honbann-cose/map_1_0509_145516")
MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/models")
LOG_DIR   = os.environ.get("LOG_DIR",   "/workspace/logs")

# --- 初期位置設定 [x, y, yaw] ---
# view_spawn.py で確認しながら調整してください
START_POSE = [-3, -3.5, 0.0]

# スタート位置のランダム化（Trueの場合、下記リストからランダムに選択）
START_POSE_RANDOMIZE = True
START_POSES = [
    [-2.08, 0.36, 0.00],  # Pose 0
    [1.80, 0.39, 1.09],  # Pose 1
    [1.16, 3.23, -2.58],  # Pose 2
    [-0.98, 3.48, 2.39],  # Pose 3



]

# モデル名に設定を反映させて管理しやすくする
MAP_NAME   = os.path.basename(MAP_PATH)
MODEL_NAME = f"ppo_f1_{MAP_NAME}_steps{TOTAL_TIMESTEPS}_arch{len(NET_ARCH)}"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
GIF_DIR    = os.path.join(PROJECT_ROOT, "gif")
GIF_PATH   = os.path.join(GIF_DIR, f"run_simulation_{MAP_NAME}_steps{TOTAL_TIMESTEPS}_arch{len(NET_ARCH)}.gif")

# 報酬計算ロジックは src/rewards.py に移動しました。
# from .rewards import calculate_reward