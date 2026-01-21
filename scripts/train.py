import gym
import f110_gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import config

class F1TenthRL(gym.Env):
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

    # 4. モデルを作成（ここでもdevice設定などを共通化できる）
    model = PPO("MlpPolicy", env, verbose=1, device='cpu')

    # 5. 学習開始
    print(f"--- 学習開始: {config.MAP_PATH} ---")
    model.learn(total_timesteps=100000)

    # 6. 保存（ここがさっきのコード！）
    model.save(config.MODEL_PATH)
    print(f"--- モデルを保存しました: {config.MODEL_PATH} ---")

if __name__ == '__main__':
    main()

import time
for _ in range(5):
    print('\a', end='', flush=True)
    time.sleep(0.3) # 0.3秒おきに鳴らす