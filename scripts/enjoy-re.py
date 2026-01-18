import gym
import f110_gym
import numpy as np
from stable_baselines3 import PPO
import imageio
import os
import matplotlib.pyplot as plt

class F1TenthRL(gym.Env):
    def __init__(self, map_path):
        super(F1TenthRL, self).__init__()
        self.env = gym.make('f110_gym:f110-v0', map=map_path, num_agents=1)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=30, shape=(1080,), dtype=np.float32)

    def reset(self):
        obs, reward, done, info = self.env.reset(np.array([[0.0, 0.0, 0.0]]))
        return obs['scans'][0].astype(np.float32)

    def step(self, action):
        # 学習時と同じ設定（クイック旋回・速度2.5）
        steer = action[0] * 0.8 
        speed = 2.5
        obs, reward, done, info = self.env.step(np.array([[steer, speed]]))
        return obs['scans'][0].astype(np.float32), reward, done, info

def main():
    map_path = '/opt/f1tenth_gym/gym/f110_gym/envs/maps/levine'
    env = F1TenthRL(map_path)
    model = PPO.load("models/ppo_f1_final")
    
    obs = env.reset()
    frames = []

    print("録画中... (広角・長距離視点)")

    for i in range(2000): # カーブの先まで撮るために延長
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        plt.figure(figsize=(7, 7))
        plt.clf()
        plt.gca().set_facecolor('#0a0a0a') # 真っ黒より少しグレーで見やすく
        
        # LiDAR描画
        angles = np.linspace(-np.pi*3/4, np.pi*3/4, 1080)
        x = obs * np.cos(angles)
        y = obs * np.sin(angles)
        
        # 遠くの点まで映るようにドットサイズを微小化(s=1)
        plt.scatter(x, y, s=1, c='cyan', alpha=0.9) 
        
        # 自車 (赤い大きな三角)
        plt.scatter(0, 0, marker='^', c='red', s=200)
        
        # --- 表示範囲を大幅に拡大 ---
        plt.xlim(-15, 15) # 左右15m
        plt.ylim(-5, 25)  # 前方25m、後方5mまで表示
        
        plt.title(f"F1Tenth AI - Step: {i}", color='white')
        plt.axis('off')
        
        fig = plt.gcf()
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close()
        
        if done:
            print(f"衝突または終了：{i}ステップ")
            break

    imageio.mimsave("run_simulation_wide.gif", frames, fps=30)
    print("完了！ 'run_simulation_wide.gif' を確認してください。")

if __name__ == '__main__':
    main()