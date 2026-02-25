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
PPO_ENT_COEF = 0.01  # エントロピー係数（探索を促進）

# --- 物理設定（マシン性能） ---
STEER_SENSITIVITY = 1.0   # ステアリングの反応速度
MIN_SPEED = 1.0            # 最低速度（これより遅くならない）
MAX_SPEED = 3.0            # 最高速度（直線で出す速度）

# --- 報酬設計の設定 ---
REWARD_COLLISION = -2000.0   # 衝突時の大きなペナルティ（より厳しく）
REWARD_SURVIVAL = 0.02      # 1ステップ生存するごとの基本報酬（少し抑える）
REWARD_FRONT_WEIGHT = 3.0   # 前方の空きスペースに対する報酬の重み
REWARD_SPEED_WEIGHT = 1.0   # 速度に対する報酬の重み
REWARD_CENTRALITY_WEIGHT = 0.5 # コース中央を走ることへの報酬
REWARD_DISTANCE_WEIGHT = 1.0   # 壁からの距離（安全マージン）への報酬

# --- パス設定 ---
#MAP_PATH = '/opt/f1tenth_gym/gym/f110_gym/envs/maps/levine' # default map 
#levine: 定番の廊下マップ
#skirk: テストコースのような形状のマップ
#berlin: 市街地コース風
#vegas: ラスベガス風
#stata_basement: 複雑な地下通路マップ
MAP_PATH = '/workspace/my_maps/my_map'  # 狭い倉庫マップ

# --- 初期位置設定 [x, y, yaw] ---
# view_spawn.py で確認しながら調整してください
START_POSE = [4.0, 4.0, 1.0] 

MODEL_DIR = "/workspace/models"
LOG_DIR = "/workspace/logs"
# モデル名に設定を反映させて管理しやすくする
MAP_NAME = os.path.basename(MAP_PATH)
MODEL_NAME = f"ppo_f1_{MAP_NAME}_steps{TOTAL_TIMESTEPS}_arch{len(NET_ARCH)}"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
# プロジェクトルートのgifディレクトリを確実に指すように修正
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIF_DIR = os.path.join(PROJECT_ROOT, "gif")
GIF_PATH = os.path.join(GIF_DIR, f"run_simulation_{MAP_NAME}_steps{TOTAL_TIMESTEPS}_arch{len(NET_ARCH)}.gif")

# --- 共通の報酬計算ロジック ---
def calculate_reward(scans, action, done, current_speed):
    """
    scans: LiDARの距離データ (1080点)
    action: AIの出力 [ステアリング, 速度]
    done: 衝突判定
    current_speed: 現在の車の速度(m/s)
    """
    if done:
        return REWARD_COLLISION
    
    # 1. 前方空間報酬（正面付近のLiDARデータ 視野を少し広げて 440〜640番）
    # 遠くまで道があるほど報酬が高い。
    front_dist = np.min(scans[440:640])
    reward = (front_dist / 30.0) * REWARD_FRONT_WEIGHT
    
    # 2. 速度報酬
    # 前方が詰まっている時は速度報酬を下げ、減速を促す
    speed_factor = current_speed / MAX_SPEED
    if front_dist < 3.0:
        reward += speed_factor * REWARD_SPEED_WEIGHT * 0.2
    else:
        reward += speed_factor * REWARD_SPEED_WEIGHT
    
    # 3. 壁接近ペナルティ（安全マージン）
    # どこであれ壁に近すぎる場合にマイナスを与える
    min_dist = np.min(scans)
    if min_dist < 0.4:  # 40cm以内は危険ゾーン
        reward -= REWARD_DISTANCE_WEIGHT * (1.0 - (min_dist / 0.4))
    
    # 4. 中央維持報酬 (Lateral Centrality)
    # 左45度(720)と右45度(360)付近の距離を比較し、バランスが良い（中央にいる）ほど報酬
    left_dist = np.min(scans[700:740])
    right_dist = np.min(scans[340:380])
    centrality = 1.0 - abs(left_dist - right_dist) / (left_dist + right_dist + 1e-6)
    reward += centrality * REWARD_CENTRALITY_WEIGHT
    
    # 5. ステアリング・安定性（条件付き）
    # 前方が開けている時（直線路）のみ、無駄な蛇行を抑制する
    if front_dist > 5.0:
        reward += (1.0 - abs(action[0])) * 0.2
        reward += REWARD_SURVIVAL
    else:
        # カーブでは生存報酬のみ与え、ステアリング自体は制限しない
        reward += REWARD_SURVIVAL
    
    return reward