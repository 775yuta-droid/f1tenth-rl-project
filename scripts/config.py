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
MAX_SPEED = 3.0            # 最高速度（直線で出す速度）

# --- 報酬設計の設定 ---
REWARD_COLLISION = -2000.0   # 衝突時の大きなペナルティ（より厳しく）
REWARD_SURVIVAL = 0.02      # 1ステップ生存するごとの基本報酬（少し抑える）
REWARD_FRONT_WEIGHT = 3.0   # 前方の空きスペースに対する報酬の重み
REWARD_SPEED_WEIGHT = 1.0   # 速度に対する報酬の重み
REWARD_CENTRALITY_WEIGHT = 0.5 # コース中央を走ることへの報酬
REWARD_DISTANCE_WEIGHT = 1.0   # 壁からの距離（安全マージン）への報酬
REWARD_PROGRESS_WEIGHT = 2.0   # 走行距離報酬（円形走行を抑制）

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
START_POSE = [3.0, 4.0, 0.0]

# スタート位置のランダム化（Trueの場合、下記リストからランダムに選択）
START_POSE_RANDOMIZE = True
START_POSES = [
    [3.0, 4.0, 0.0],   # デフォルト位置
    [3.0, 4.0, 1.0],   # 少し左向き
    [3.0, 4.0, 5.0],   # 右向き
    [5.0, 4.0, 1.5],   # 少し北側
    [2.0, 5.5, 0.5],   # 少し東側
]

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
def calculate_reward(scans, action, done, current_speed, prev_x=0.0, prev_y=0.0, cur_x=0.0, cur_y=0.0):
    """
    scans: LiDARの距離データ (1080点)
    action: AIの出力 [ステアリング, 速度]
    done: 衝突判定
    current_speed: 現在の車の速度(m/s)
    prev_x, prev_y: 前ステップの位置
    cur_x, cur_y: 現在の位置
    """
    if done:
        return REWARD_COLLISION

    # 1. 前方空間報酬（視野を±45度相当まで拡大: 350～730番）
    # これによりコーナーの山を早めに認識できる
    front_dist = np.min(scans[350:730])
    reward = (front_dist / 30.0) * REWARD_FRONT_WEIGHT

    # 2. 速度報酬
    # 前方が広い時は高速に、コーナーでは減速を促す（逆インセンティブ強化）
    speed_factor = current_speed / MAX_SPEED
    if front_dist < 2.0:
        # コーナー直前: 速度報酬をゼロに▼減速すること自体が有利に
        reward += speed_factor * REWARD_SPEED_WEIGHT * 0.0
    elif front_dist < 4.0:
        reward += speed_factor * REWARD_SPEED_WEIGHT * 0.2
    else:
        reward += speed_factor * REWARD_SPEED_WEIGHT

    # 3. 壁接近ペナルティ（安全マージン）
    # 閘値を 1.0m に引き上げて壁ギリギリ走行を抑制
    min_dist = np.min(scans)
    if min_dist < 1.0:
        reward -= REWARD_DISTANCE_WEIGHT * (1.0 - (min_dist / 1.0))

    # 4. 中央維持報酬 (Lateral Centrality)
    left_dist = np.min(scans[700:740])
    right_dist = np.min(scans[340:380])
    centrality = 1.0 - abs(left_dist - right_dist) / (left_dist + right_dist + 1e-6)
    reward += centrality * REWARD_CENTRALITY_WEIGHT

    # 5. 走行距離報酬（円形走行抑制）
    progress = np.sqrt((cur_x - prev_x) ** 2 + (cur_y - prev_y) ** 2)
    reward += progress * REWARD_PROGRESS_WEIGHT

    # 6. ステアリング・安定性（条件付き）
    if front_dist > 5.0:
        reward += (1.0 - abs(action[0])) * 0.2
        reward += REWARD_SURVIVAL
    else:
        reward += REWARD_SURVIVAL
    
    return reward