import numpy as np
import os
import multiprocessing
from src.profiles import PROFILES

# --- デバイス設定 ---
DEVICE = "cpu"

# --- 学習環境プロファイル ---
_profile_name = os.environ.get("TRAINING_PROFILE", "auto")
_profile = PROFILES.get(_profile_name, PROFILES["auto"])

_cpu_count = multiprocessing.cpu_count()
_profile_threads = _profile["torch_num_threads"]
if _profile_threads is None:
    _profile_threads = 2 if _cpu_count >= 2 else 1
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", _profile_threads))

# --- 学習ハイパーパラメータ ---
TOTAL_TIMESTEPS = 10000000
LEARNING_RATE = 5e-5

# --- ネットワーク構造 ---
NET_ARCH = [128, 128]

# --- 後方互換性用（現在のスクリプトでエラーを出さないため） ---
USE_CNN_POLICY = False
LIDAR_BEAMS = 1440
CONTROL_HZ = 40
SIM_TIMESTEP = 1.0 / CONTROL_HZ
INCLUDE_EXTRA_FEATURES = False

# --- 観測空間の工夫 ---
LIDAR_DOWNSAMPLE_FACTOR = 10
FRAME_STACK = 4
FRAME_SKIP = 3  # F1TenthRL環境が要求するため付与
N_ENVS = 8
INCLUDE_VEHICLE_STATE = True
INCLUDE_LIDAR_RESIDUAL = False

# --- 正規化設定 ---
NORMALIZE_OBSERVATIONS = True
LIDAR_MAX_RANGE = 30.0
# Calibrated statistics based on 10000 random steps
LIDAR_MEAN = 4.869
LIDAR_STD = 3.577
LIDAR_RESIDUAL_MEAN = -0.008
LIDAR_RESIDUAL_STD = 0.084
VEHICLE_STATE_MEAN = np.array([0.574, -0.010])  # [vel, steer]
VEHICLE_STATE_STD = np.array([0.096, 0.122])

# --- PPO 探索設定 ---
PPO_ENT_COEF = 0.01

# --- 物理設定（マシン性能） ---
STEER_SENSITIVITY = 1.0
MIN_SPEED = 0.3
MAX_SPEED = 2.5

# --- マシン寸法 ---
CAR_LENGTH = 0.465
CAR_WIDTH = 0.19

# --- 報酬設計の設定 ---
REWARD_COLLISION = -200.0
REWARD_SURVIVAL  = 0.2
REWARD_FRONT_WEIGHT = 3.0
REWARD_SPEED_WEIGHT = 1.0
REWARD_SAFETY_WEIGHT = 0.8
REWARD_DISTANCE_WEIGHT = 1.0
REWARD_PROGRESS_WEIGHT = 2.0

# --- パス設定 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_PATH  = os.environ.get("MAP_PATH",  "/workspace/my_maps/my_map")
MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/models")
LOG_DIR   = os.environ.get("LOG_DIR",   "/workspace/logs")

# --- 初期位置設定 [x, y, yaw] ---
START_POSE = [2.5, 4.0, 0.0]
START_POSE_RANDOMIZE = True
START_POSES = [
    [1.5, 3.5,  0.5],
    [3.0, 5.0,  2.5],
]

MAP_NAME   = os.path.basename(MAP_PATH)
MODEL_NAME = f"ppo_f1_{MAP_NAME}_steps{TOTAL_TIMESTEPS}_arch{len(NET_ARCH)}"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
GIF_DIR    = os.path.join(PROJECT_ROOT, "gif")
GIF_PATH   = os.path.join(GIF_DIR, f"run_simulation_{MAP_NAME}_steps{TOTAL_TIMESTEPS}_arch{len(NET_ARCH)}.gif")
