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
TOTAL_TIMESTEPS = 3000000
LEARNING_RATE = 3e-5  # 互換性のため残存。TD3学習率は TD3_LEARNING_RATE で管理。

# --- TD3 ハイパーパラメータ ---
TD3_LEARNING_RATE   = 1e-4     # 先生推奨値。SACより小さめが安定。
TD3_BATCH_SIZE      = 512      # GPU効率化のため大きめ
TD3_BUFFER_SIZE     = 500_000  # リプレイバッファサイズ。N_ENVS=1フォーカスから多めに確保。
TD3_LEARNING_STARTS = 10_000   # 最初はランダム行動で経験を貯める
TD3_ACTION_NOISE_SIGMA = 0.1   # 探索ノイズ。最大アクション幅(±1.0)の10%。

# --- 非推奨: PPO固有パラメータ（参照用に残存） ---
PPO_BATCH_SIZE = 512
PPO_N_STEPS = 1024  # 2048 -> 1024: 最初の更新を早く行い、スピン固定化を防ぐ

# --- ネットワーク構造 ---
# EXP-47: [128, 128] -> [256, 256] (features_dim=512に対するボトルネック解消。情報の取りこぼしを防ぐ)
NET_ARCH = [256, 256]

# --- 観測空間の工夫 ---
LIDAR_BEAMS = 1440             # シミュレータの全周ビーム数 (360°分)
# 270°分を1080点とするため、360°では 1080 * 360 / 270 = 1440点 となる (0.25°刻み)
LIDAR_DOWNSAMPLE_FACTOR = 5   # EXP-39: 解像度を2倍に(10->5)
FRAME_STACK = 2                # スタックするフレーム数
FRAME_SKIP = 4                 # EXP-44: 1 -> 4 (視野を 0.1s -> 0.4s に拡大し、CNNが動きを捉えやすくする)
N_ENVS = 1                     # TD3はオフポリシーのためリプレイバッファで学習。並列化不要。
INCLUDE_VEHICLE_STATE = True  # 速度とステアリング角を観測に含める
INCLUDE_LIDAR_RESIDUAL = False # ΔLiDAR は行動安定に寄与 (EXP-15: ノイズ排除のため無効化)
INCLUDE_EXTRA_FEATURES = True  # EXP-49: カーブ方向推定のため lr_asymmetry 含む3特徴量を有効化
INCLUDE_RACING_LINE   = True   # レーシングライン観測: [cte, heading_err, curvature, progress] 4次元
INCLUDE_ACTION_HISTORY = True  # 前ステップ行動履歴: [prev_steer, prev_speed] 2次元（滑らかな操作を促進）

# --- 正規化設定 ---
# development-plan.md 推奨: Z-score ではなく 0-1 反転方式を採用
#   lidar_norm = 1.0 - clip(lidar, 0, MAX_RANGE) / MAX_RANGE
#   → 近い壁 = 1.0, 遠い空間 = 0.0  (NaN/inf を安全にクリップ後に適用)
NORMALIZE_OBSERVATIONS = True
LIDAR_MAX_RANGE = 30.0         # クリッピング上限 (m)

# --- CNN ポリシー設定 ---
# True: Conv1DLidarExtractor + MlpPolicy, False: 従来の MlpPolicy (MLP のみ)
USE_CNN_POLICY = False

# --- PPO 探索設定（非推奨: TD3ではaction_noiseで探索を制御） ---
PPO_ENT_COEF = 0.03

# --- 物理設定（マシン性能） ---
CONTROL_HZ = 40            # 実機LiDARに合わせた制御周波数 (40Hz)
ACTION_REPEAT = 4          # EXP-44: 1 -> 4 (10Hz制御に落とし、低速時の挙動を安定化)
SIM_TIMESTEP = 1.0 / CONTROL_HZ
STEER_SENSITIVITY = 1.0    # EXP-35: 1.3 -> 1.0 に復帰 (元の感度に戻し、AIの運転感覚の狂いを解消)
MIN_SPEED = float(os.environ.get("MIN_SPEED", "0.5"))   # 低速停止旋回を抑制しつつ、短時間での前進学習を促す
MAX_SPEED = float(os.environ.get("MAX_SPEED", "1.2"))   # [Fix-A2] 1.0 → 2.0: 報酬関数の max_speed と整合し、カーブ報酬を正常化

# --- マシン寸法 ---
CAR_LENGTH = 0.465
CAR_WIDTH = 0.19           # EXP-35: 0.23 -> 0.19 に復帰 (太さを元に戻し、物理的に狭いコースを曲がれるようにする)

# --- 残差強化学習 (Residual RL) 設定 ---
USE_RESIDUAL_RL = True    # True: 古典制御(Pure Pursuit) + RL補正, False: 通常のRL
                            # [Fix-V4] スピン原因の切り分けのため一時無効化
RESIDUAL_STEER_SCALE = 0.2 # [Fix-B] 0.1→0.2: ステア補正幅を拡大（最大ステア21%→ターン補正可)
RESIDUAL_SPEED_SCALE = 1.0 # 速度補正幅を半分に減らし、速度変動を穏やかに
PURE_PURSUIT_LOOKAHEAD = 1.5 # [Fix-C] 2.5→1.5: 犭いマップの急カーブで目標点精度を改善

# --- 報酬設計の設定 ---
REWARD_COLLISION = -200.0
REWARD_SURVIVAL  = 0.1
REWARD_FRONT_WEIGHT = 0.0
REWARD_SPEED_WEIGHT = 2.0
REWARD_SAFETY_WEIGHT = 0.8
REWARD_DISTANCE_WEIGHT = 1.0
REWARD_PROGRESS_WEIGHT = 10.0
REWARD_CURVE_WEIGHT    = 1.5
REWARD_LINE_WEIGHT     = 0.5
REWARD_SMOOTH_WEIGHT   = 0.1
YAW_RATE_PENALTY_WEIGHT = 1.5

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

MAP_PATH  = os.environ.get("MAP_PATH",  "/workspace/my_maps/magp-map/map_3_0525_103844")
MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/models")
LOG_DIR   = os.environ.get("LOG_DIR",   "/workspace/logs")

# レーシングライン CSV のパス（MAP_PATHから自動導出、環境変数で上書き可）
# 生成コマンド: python3 scripts/utils/generate_centerline.py
RACING_LINE_PATH = os.environ.get("RACING_LINE_PATH", MAP_PATH + "_centerline.csv")

# --- 初期位置設定 [x, y, yaw] ---
# view_spawn.py で確認しながら調整してください
START_POSE = [0.39, -0.61, 0.15]

# スタート位置のランダム化（Trueの場合、下記リストからランダムに選択）
START_POSE_RANDOMIZE = True
START_POSES = [
    [0.39, -0.61, 0.15],
    [1.34, -0.85, -0.46],
    [1.14, -2.34, -2.55],
    [0.33, -4.41, -1.86],
    [-0.73, -5.60, -1.73],
    [-2.14, -10.20, 1.46],
    [-1.21, -11.23, 3.08],
    [-1.87, -9.10, 1.42],
    [-2.03, -7.70, 1.82],
]

# モデル名に設定を反映させて管理しやすくする
MAP_NAME   = os.path.basename(MAP_PATH)
MODEL_NAME = f"td3_f1_{MAP_NAME}_steps{TOTAL_TIMESTEPS}_arch{len(NET_ARCH)}"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
GIF_DIR    = os.path.join(PROJECT_ROOT, "gif")
GIF_PATH   = os.path.join(GIF_DIR, f"run_simulation_{MAP_NAME}_steps{TOTAL_TIMESTEPS}_arch{len(NET_ARCH)}.gif")

# 報酬計算ロジックは src/rewards.py に移動しました。
# from .rewards import calculate_reward

# --- Human Demo / Behavioral Cloning ---
# 実機デモ録画 → convert_demo.py → pretrain_bc.py → train.py --resume のパイプライン
DEMO_DIR       = os.path.join(PROJECT_ROOT, "demos")
BC_MODEL_PATH  = os.path.join(MODEL_DIR, "bc_pretrained")
BC_EPOCHS      = 500          # BC 学習エポック数
BC_BATCH_SIZE  = 256          # BC バッチサイズ
BC_LR          = 3e-4         # BC 学習率
REAL_STEER_MAX = 0.4189       # 実機最大ステア角 [rad] (Futaba T4PM / Hokuyo)