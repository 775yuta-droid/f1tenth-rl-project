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
    random_seed =  range(0, 10000)

    def reset(self):
        # コースのスタート位置をリセット
        obs, reward, done, info = self.env.reset(np.array([[0.0, np.random.choice(self.random_seed), 0.0]]))
        return obs['scans'][0].astype(np.float32)

    def step(self, action):
        # AIの出力を実際の操作量に変換
        steer = action[0] * 0.4 
        speed = 3.5  # 止まらないように固定速度を設定
        
        obs, reward, done, info = self.env.step(np.array([[steer, speed]]))
        scans = obs['scans'][0]

        
        
        # 報酬設計: 空間探索を促進する工夫
    def step(self, action):
        # 1. ステアリング感度を0.4から0.8に強化（よりクイックに曲がる）
        steer = action[0] * 0.8 
        # 2. 速度を3.5から2.5に落とし、確実に曲がれるようにする
        speed = 2.5  
        
        obs, reward, done, info = self.env.step(np.array([[steer, speed]]))
        scans = obs['scans'][0]

        if done:
            # 衝突ペナルティをさらに重く変更
            reward = -200.0
        else:
            # 基本の前進報酬
            reward = 1.0 
            
            # 2. 【重要】「道幅」を考慮した前方報酬
            # 前方 central 20度の範囲の「最小距離」を見ることで、
            # 狭い隙間に騙されにくくする
            center_scans = scans[530:550] 
            min_front_dist = np.min(center_scans)
            
            # 広い道なら高い報酬、狭い隙間なら低い報酬
            reward += min_front_dist * 1.5

            # 3. 左右のバランス報酬（路地の入口を避ける）
            # 左右に極端に近い壁がある場合はマイナス
            if np.min(scans[440:640]) < 0.5: # 目の前に壁が迫ったら
                reward -= 5.0

        return obs['scans'][0].astype(np.float32), reward, done, info

def main():
    # 正しい階層（gymが2回重なっている方）かつ 拡張子なしで指定
    map_path = '/opt/f1tenth_gym/gym/f110_gym/envs/maps/levine'
    if not os.path.exists("models"):
        os.makedirs("models")

    env = F1TenthRL(map_path)
    env = DummyVecEnv([lambda: env])

    # 学習設定 (device='cpu' で警告を抑制)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, device='cpu')

    print("--- 空間探索モードで学習を開始します (目標: 万回) ---")
    # 学習回数
    model.learn(total_timesteps=100000)
    
    # モデルの保存
    model.save("../models/ppo_f1_final")
    print("--- 学習完了！モデルを保存しました ---")

if __name__ == '__main__':
    main()

import time
for _ in range(5):
    print('\a', end='', flush=True)
    time.sleep(0.3) # 0.3秒おきに鳴らす