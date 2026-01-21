import numpy as np

# --- デバイス設定  ---
# "cuda" : GPUを使用 (NVIDIAのGPUが必要)
# "cpu"  : CPUを使用
# "auto" : 利用可能ならGPU、なければCPU (Stable Baselines3の機能)
#cpuで動かすことを推奨
DEVICE = "cpu"

# --- 学習ハイパーパラメータ ---
TOTAL_TIMESTEPS = 100000  # 学習回数
LEARNING_RATE = 3e-4      # 学習率

# --- ネットワーク構造の設定 ---
# [64, 64] は 64ユニットの層が2つ。 [256, 256, 256] とすればより深く複雑になります。
NET_ARCH = [64, 64]

# --- 物理設定（マシン性能） ---
STEER_SENSITIVITY = 0.8  # ステアリングの強さ
MAX_SPEED = 2.5          # 走行速度

# --- 報酬設計の設定 ---
REWARD_COLLISION = -500.0   # 衝突時のマイナス
REWARD_SURVIVAL = 0.1       # 1ステップ生存ごとの基本報酬
REWARD_FRONT_WEIGHT = 2.0   # 前方空間報酬の重み

# --- パス設定 ---
MAP_PATH = '/opt/f1tenth_gym/gym/f110_gym/envs/maps/levine'
MODEL_DIR = "../models"
MODEL_NAME = f"ppo_f1_steps{TOTAL_TIMESTEPS}_arch{len(NET_ARCH)}"
MODEL_PATH = f"{MODEL_DIR}/{MODEL_NAME}"
GIF_PATH = "../gif/run_simulation_wide.gif"

# --- 共通の報酬計算ロジック ---
def calculate_reward(scans, action, done):
    if done:
        return REWARD_COLLISION
    
    # 基本報酬
    reward = REWARD_SURVIVAL
    
    # 前方空間報酬 (正規化)
    center_dist = np.min(scans[500:580])
    reward += (center_dist / 30.0) * REWARD_FRONT_WEIGHT
    
    # 直進ボーナス（ハンドルを切っていないほどプラス）
    reward += (1.0 - abs(action[0])) * 0.2
    
    return reward