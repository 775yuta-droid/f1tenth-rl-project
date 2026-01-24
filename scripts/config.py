# import numpy as np

# # --- デバイス設定  ---
# # "cuda" : GPUを使用 (NVIDIAのGPUが必要)
# # "cpu"  : CPUを使用
# # "auto" : 利用可能ならGPU、なければCPU (Stable Baselines3の機能)
# #cpuで動かすことを推奨
# DEVICE = "cpu"

# # --- 学習ハイパーパラメータ ---
# TOTAL_TIMESTEPS = 500000  # 学習回数
# LEARNING_RATE = 3e-4      # 学習率

# # --- ネットワーク構造の設定 ---
# # [64, 64] は 64ユニットの層が2つ。 [256, 256, 256] とすればより深く複雑になります。
# NET_ARCH = [64, 64]

# # --- 物理設定（マシン性能） ---
# STEER_SENSITIVITY = 0.8  # ステアリングの強さ
# MIN_SPEED = 1.0  # 最低速度（遅すぎると学習が停滞するので1.0程度がおすすめ）
# MAX_SPEED = 4.0  # 最高速度
# STEER_SENSITIVITY = 0.8


# # --- 報酬設計の設定 ---
# REWARD_COLLISION = -500.0   # 衝突時のマイナス
# REWARD_SURVIVAL = 0.1       # 1ステップ生存ごとの基本報酬
# REWARD_FRONT_WEIGHT = 2.0   # 前方空間報酬の重み

# # --- パス設定 ---
# MAP_PATH = '/opt/f1tenth_gym/gym/f110_gym/envs/maps/levine'
# MODEL_DIR = "../models"
# MODEL_NAME = f"ppo_f1_steps{TOTAL_TIMESTEPS}_arch{len(NET_ARCH)}"
# MODEL_PATH = f"{MODEL_DIR}/{MODEL_NAME}"
# GIF_PATH = "../gif/run_simulation_wide.gif"

# # --- 共通の報酬計算ロジック ---
# def calculate_reward(scans, action, done):
#     if done:
#         return REWARD_COLLISION
    
#     # 基本報酬
#     reward = REWARD_SURVIVAL

#     # 速度に応じた報酬 (action[1]が高い＝速いほど得点)
#     speed_reward = (action[1] + 1.0) * 2.0
    
#     # 前方空間報酬 (正規化) 
#     center_dist = np.min(scans[500:580])
#     reward += (center_dist / 30.0) * REWARD_FRONT_WEIGHT
    
#     # 直進ボーナス（ハンドルを切っていないほどプラス）
#     reward += (1.0 - abs(action[0])) * 0.2
    
#     return reward

import numpy as np
import os

# --- デバイス設定 ---
# 互換性重視のため CPU を指定
DEVICE = "cpu"  # "cpu", "cuda", "auto" から選択可能

# --- 学習ハイパーパラメータ ---
# ステアリング+速度の2次元学習は時間がかかるため、300,000〜500,000を推奨
TOTAL_TIMESTEPS = 100000000
LEARNING_RATE = 3e-4

# --- ネットワーク構造 ---
# 複雑な判断（加減速）をさせるため、少し深めの [128, 128] に設定
NET_ARCH = [128, 128]

# --- 物理設定（マシン性能） ---
STEER_SENSITIVITY = 0.8   # ステアリングの反応速度
MIN_SPEED = 0.7            # 最低速度（これより遅くならない）
MAX_SPEED = 3.0            # 最高速度（直線で出す速度）

# --- 報酬設計の設定 ---
REWARD_COLLISION = -500.0   # 衝突時の大きなペナルティ
REWARD_SURVIVAL = 0.1       # 1ステップ生存するごとの基本報酬
REWARD_FRONT_WEIGHT = 2.5   # 前方の空きスペースに対する報酬の重み
REWARD_SPEED_WEIGHT = 0.8   # 「速く走る」ことに対する報酬の重み

# --- パス設定 ---
# MAP_PATH = '/opt/f1tenth_gym/gym/f110_gym/envs/maps/levine' # default map
MAP_PATH = '/workspace/my_maps/my_map'  # 狭い倉庫マップ
MODEL_DIR = "/workspace/models"
# モデル名に設定を反映させて管理しやすくする
MODEL_NAME = f"ppo_f1_custom_map_steps{TOTAL_TIMESTEPS}_arch{len(NET_ARCH)}"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
GIF_PATH = "../gif/run_simulation_wide.gif"

# --- 共通の報酬計算ロジック ---
def calculate_reward(scans, action, done, current_speed):
    """
    scans: LiDARの距離データ
    action: AIの出力 [ステアリング, 速度]
    done: 衝突判定
    current_speed: 現在の車の速度(m/s)
    """
    if done:
        return REWARD_COLLISION
    
    # 1. 生存報酬
    reward = REWARD_SURVIVAL
    
    # 2. 前方空間報酬（正面付近のLiDARデータ 500〜580番 を使用）
    # 遠くまで道があるほど報酬が高い。30mを最大値として正規化。
    center_dist = np.min(scans[500:580])
    reward += (center_dist / 30.0) * REWARD_FRONT_WEIGHT
    
    # 3. 速度報酬（可変速度のキモ）
    # MAX_SPEEDに近いほど高い報酬を与えることで、AIに加速を促す
    reward += (current_speed / MAX_SPEED) * REWARD_SPEED_WEIGHT
    
    # 4. 直進・安定性ボーナス
    # ハンドルを大きく切っていない（abs(action[0])が小さい）ほどプラス
    reward += (1.0 - abs(action[0])) * 0.2
    
    return reward

# def calculate_reward(scans, action, done, current_speed):
#     if done:
#         return REWARD_COLLISION
    
#     reward = REWARD_SURVIVAL
    
#     # --- 1. 前方空間報酬（正面 500〜580） ---
#     center_dist = np.min(scans[500:580])
#     reward += (center_dist / 30.0) * REWARD_FRONT_WEIGHT
    
#     # --- 2. 路地回避のための「中央維持」報酬（追加） ---
#     # 左側(例:800番付近)と右側(例:280番付近)の距離を比較
#     left_dist = np.mean(scans[750:850])
#     right_dist = np.mean(scans[230:330])
    
#     # 左右の距離の差が小さいほど「コース中央」にいるとみなす
#     # 差が大きい＝どちらかの壁に寄っている（または路地が片側に見えている）
#     diff = abs(left_dist - right_dist)
#     reward -= diff * 0.1  # ペナルティとして引く
    
#     # --- 3. 速度報酬 ---
#     reward += (current_speed / MAX_SPEED) * REWARD_SPEED_WEIGHT
    
#     # --- 4. 安定性ボーナス ---
#     reward += (1.0 - abs(action[0])) * 0.2
    
#     return reward