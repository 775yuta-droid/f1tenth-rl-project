import numpy as np
import os

# --- デバイス設定 ---
# 互換性重視のため CPU を指定
DEVICE = "cpu"  # "cpu", "cuda", "auto" から選択可能

# --- 学習ハイパーパラメータ ---
# ステアリング+速度の2次元学習は時間がかかるため、300,000〜500,000を推奨
TOTAL_TIMESTEPS = 1500000
LEARNING_RATE = 1e-4

# --- ネットワーク構造 ---
# 複雑な判断（加減速）をさせるため、少し深めの [128, 128] に設定
NET_ARCH = [128, 128]

# --- 観測空間の工夫 ---
LIDAR_DOWNSAMPLE_FACTOR = 2   # 1080 -> 540次元（残差処理のため高解像度を維持）
INCLUDE_VEHICLE_STATE = True  # 速度とステアリング角を観測に含める
INCLUDE_LIDAR_RESIDUAL = True # 前ステップとのLiDAR差分（ΔLiDAR）を観測に含める

# --- 正規化設定 ---
NORMALIZE_OBSERVATIONS = True
# Calibrated statistics based on 10000 random steps
LIDAR_MEAN = 4.555
LIDAR_STD = 3.894
LIDAR_RESIDUAL_MEAN = -0.012
LIDAR_RESIDUAL_STD = 0.096
VEHICLE_STATE_MEAN = np.array([0.528, 0.003])  # [vel, steer]
VEHICLE_STATE_STD = np.array([0.107, 0.115])

# --- PPO 探索設定 ---
PPO_ENT_COEF = 0.005  # エントロピー係数（探索を促進、少なめ）

# --- 物理設定（マシン性能） ---
STEER_SENSITIVITY = 1.0   # ステアリングの反応速度
MIN_SPEED = 1.0            # 最低速度（これより遅くならない）
MAX_SPEED = 2.5            # 最高速度（3.0→コーナーで安全な速度に下げ）

# --- 報酬設計の設定 ---
REWARD_COLLISION = -2000.0   # 衝突時の大きなペナルティ（より厳しく）
REWARD_SURVIVAL = 0.02      # 1ステップ生存するごとの基本報酬（少し抑える）
REWARD_FRONT_WEIGHT = 3.0   # 前方の空きスペースに対する報酬の重み
REWARD_SPEED_WEIGHT = 1.0   # 速度に対する報酬の重み
REWARD_CENTRALITY_WEIGHT = 0.5 # コース中央を走ることへの報酬
REWARD_DISTANCE_WEIGHT = 1.0   # 壁からの距離（安全マージン）への報酬
REWARD_PROGRESS_WEIGHT = 2.0   # 走行距離報酬（円形走行を抑制）

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
START_POSE = [3.0, 4.0, 0.0]

# スタート位置のランダム化（Trueの場合、下記リストからランダムに選択）
START_POSE_RANDOMIZE = True
START_POSES = [
    [3.0, 4.0,  0.0],
    [3.0, 5.0,  0.5],
    [3.0, 5.0,  2.5],
    [3.0, 4.0,  3.14],
    [0.7, 5.0, -1.0],
    [5.0, 4.5, -2.0],
]

# モデル名に設定を反映させて管理しやすくする
MAP_NAME   = os.path.basename(MAP_PATH)
MODEL_NAME = f"ppo_f1_{MAP_NAME}_steps{TOTAL_TIMESTEPS}_arch{len(NET_ARCH)}"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
GIF_DIR    = os.path.join(PROJECT_ROOT, "gif")
GIF_PATH   = os.path.join(GIF_DIR, f"run_simulation_{MAP_NAME}_steps{TOTAL_TIMESTEPS}_arch{len(NET_ARCH)}.gif")

# 報酬計算ロジックは src/rewards.py に移動しました。
# 後方互換のため、このファイルから直接呼び出さないでください。
def calculate_reward(scans, action, done, current_speed, prev_x=0.0, prev_y=0.0, cur_x=0.0, cur_y=0.0):
    """Deprecated: src/rewards.calculate_reward() を使用してください。"""
    import warnings
    warnings.warn(
        "config.calculate_reward() は非推奨です。src/rewards.calculate_reward() を使用してください。",
        DeprecationWarning, stacklevel=2
    )
    from src.rewards import calculate_reward as _calculate_reward
    return _calculate_reward(scans, action, done, current_speed, prev_x, prev_y, cur_x, cur_y)