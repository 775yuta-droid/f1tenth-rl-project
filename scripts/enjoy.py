import gym
import f110_gym
import numpy as np
from stable_baselines3 import PPO
import os
import imageio
import matplotlib.pyplot as plt

class F1TenthRL(gym.Env):
    def __init__(self, map_path):
        super(F1TenthRL, self).__init__()
        self.env = gym.make('f110_gym:f110-v0', map=map_path, num_agents=1)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=30, shape=(1080,), dtype=np.float32)

    def reset(self):
        obs, reward, done, info = self.env.reset(np.array([[0.0, 0.0, 0.0]]))
        return obs['scans'][0].astype(np.float32)

    def step(self, action):
        steer = action[0] * 0.4 
        speed = 2.0
        obs, reward, done, info = self.env.step(np.array([[steer, speed]]))
        return obs['scans'][0].astype(np.float32), reward, done, info

def main():
    print("--- AI走行テスト（LiDARプロット版）を開始します ---")
    map_path = '/opt/f1tenth_gym/gym/f110_gym/envs/maps/levine'
    output_gif_path = "run_simulation.gif"

    env = F1TenthRL(map_path)
    model = PPO.load("../models/ppo_f1_final" , device='cpu')#互換性問題の解決のためcpu指定
    
    obs = env.reset()
    frames = []

    # 背景画像なしでグラフを初期化
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')

    for i in range(1200):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # --- 描画処理 ---
        ax.clear()
        ax.set_facecolor('black')
        
        # スキャンデータの座標変換 (極座標 -> 直交座標)
        scans = obs
        # F1TenthのLiDARは -2.35rad から 2.35rad の範囲
        angles = np.linspace(-2.35, 2.35, 1080)
        
        # 車両を中心とした相対座標
        x = scans * np.cos(angles)
        y = scans * np.sin(angles)
        
        # 壁の点をプロット
        ax.scatter(y, x, s=2, c='cyan', alpha=0.8) # 進行方向を上にするため x と y を入れ替え
        
        # 車両の位置（原点）に自車をプロット
        ax.scatter(0, 0, s=50, c='red', marker='^') 
        
        ax.set_xlim([-10, 10]) # 左右10mの範囲
        ax.set_ylim([-2, 15])  # 前方15m, 後方2mの範囲
        ax.set_title(f"AI Vision - Step: {i}", color='white')
        ax.axis('off')
        
        # グラフを画像に変換
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)

        if i % 20 == 0:
            print(f"ステップ {i}: 走行中...")
        if done:
            print("衝突しました")
            break

    plt.close(fig)

    if frames:
        print(f"{len(frames)}枚の画像からGIFを作成中...")
        imageio.mimsave(output_gif_path, frames, fps=15)
        print(f"作成完了: {output_gif_path}")

if __name__ == '__main__':
    main()