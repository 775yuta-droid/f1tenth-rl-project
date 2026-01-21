import numpy as np

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
MODEL_NAME = "ppo_f1_final"
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