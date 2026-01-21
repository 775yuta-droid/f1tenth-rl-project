import gym
import f110_gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import config

class F1TenthRL(gym.Env):
    def __init__(self, map_path):
        super(F1TenthRL, self).__init__()
        # シミュレータの初期化（これがないと self.env が使えません）
        self.env = gym.make('f110_gym:f110-v0', map=map_path, num_agents=1)
        
        # アクションと観測空間の定義
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=30, shape=(1080,), dtype=np.float32)

    def reset(self):
        # 必須：リセット処理
        obs, reward, done, info = self.env.reset(np.array([[0.0, 0.0, 0.0]]))
        return obs['scans'][0].astype(np.float32)
    def step(self, action):
        # configから設定を読み込む
        steer = action[0] * config.STEER_SENSITIVITY
        speed = config.MAX_SPEED
        
        obs, _, done, info = self.env.step(np.array([[steer, speed]]))
        scans = obs['scans'][0]
        
        # 報酬計算も共通関数を呼び出すだけ！
        reward = config.calculate_reward(scans, action, done)
        return scans.astype(np.float32), reward, done, info

def main():
    # 1. 司令塔から設定を読み込む
    map_path = config.MAP_PATH
    
    # 2. 保存先フォルダがあるかチェック（これもconfigから！）
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)

    # 3. 環境をセットアップ
    env = F1TenthRL(map_path)
    env = DummyVecEnv([lambda: env])

    # 4. モデルを作成
    policy_kwargs = dict(net_arch=config.NET_ARCH) # ネットワーク構造を指定
        
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=config.LEARNING_RATE,
        policy_kwargs=policy_kwargs,
        verbose=1, 
        device=config.DEVICE
    )

    # 5. 学習開始
    print(f"--- 学習開始: {config.MAP_PATH} ---")
    model.learn(total_timesteps=100000)

    # 6. 保存
    model.save(config.MODEL_PATH)
    print(f"--- モデルを保存しました: {config.MODEL_PATH} ---")

if __name__ == '__main__':
    main()