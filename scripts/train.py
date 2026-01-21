import gym
import f110_gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

class F1TenthRL(gym.Env):
    def __init__(self, map_path):
        super(F1TenthRL, self).__init__()
        self.env = gym.make('f110_gym:f110-v0', map=map_path, num_agents=1)
        # アクション: ステアリングのみ (速度は固定)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # 観測: LiDARデータ
        self.observation_space = gym.spaces.Box(low=0, high=30, shape=(1080,), dtype=np.float32)

    # 初期位置をランダムに設定
    #random_seed =  range(0, 10000)

    def reset(self):
        # コースのスタート位置をリセット
        obs, reward, done, info = self.env.reset(np.array([[0.0, 0.0, 0.0]]))
        return obs['scans'][0].astype(np.float32)

        
        # 報酬設計: 空間探索を促進する工夫
    def step(self, action):
        steer = action[0] * 0.8 
        speed = 2.5  
        
        obs, reward, done, info = self.env.step(np.array([[steer, speed]]))
        
        # 正規化（0-30m を 0-1 に）
        scans = np.clip(obs['scans'][0], 0, 30) / 30.0

        if done:
            # 衝突ペナルティをさらに重く変更
            reward = -200.0

        else:
                    # 前方の空きスペースを報酬にする
                    center_dist = np.min(scans[530:550])
                    reward = 1.0 + (center_dist * 5.0) # 正規化したので係数を大きくする

        return scans.astype(np.float32), reward, done, info 

def main():
    # 正しい階層（gymが2回重なっている方）かつ 拡張子なしで指定
    map_path = '/opt/f1tenth_gym/gym/f110_gym/envs/maps/levine'
    if not os.path.exists("models"):
        os.makedirs("models")

    env = F1TenthRL(map_path)
    env = DummyVecEnv([lambda: env])

    # 学習設定 (device='cpu' で警告を抑制)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, device='cpu')

    # 変数に学習回数を設定
    total_steps = 100000

    print(f"--- 空間探索モードで学習を開始します (目標: {total_steps} 回) ---")

    # 学習回数に変数を渡す
    model.learn(total_timesteps=total_steps)

    
    # モデルの保存
    model.save("../models/ppo_f1_final")
    print("--- 学習完了！モデルを保存しました ---")

if __name__ == '__main__':
    main()

import time
for _ in range(5):
    print('\a', end='', flush=True)
    time.sleep(0.3) # 0.3秒おきに鳴らす