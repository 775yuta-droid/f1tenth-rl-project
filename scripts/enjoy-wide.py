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
        # アクションスペースはtrain.pyと一致させる
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=30, shape=(1080,), dtype=np.float32)

    def reset(self):
        obs, reward, done, info = self.env.reset(np.array([[0.0, 0.0, 0.0]]))
        return obs['scans'][0].astype(np.float32)

    def step(self, action):
        # 最新の学習設定（ステアリング感度0.8、速度2.5）に合わせるのが推奨ですが、
        # ここでは元のコードの挙動をベースにしています。
        steer = action[0] * 0.8  # クイック旋回設定
        speed = 2.5              # 安定速度
        obs, reward, done, info = self.env.step(np.array([[steer, speed]]))
        return obs['scans'][0].astype(np.float32), reward, done, info

def main():
    print("--- AI走行テスト（広角LiDARプロット版）を開始します ---")
    map_path = '/opt/f1tenth_gym/gym/f110_gym/envs/maps/levine'
    output_gif_path = "run_simulation_wide.gif"

    env = F1TenthRL(map_path)
    model = PPO.load("models/ppo_f1_final")
    
    obs = env.reset()
    frames = []

    # 画面サイズを少し大きくして解像度を上げる
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')

    for i in range(2000): # 長い直線やコーナーも収まるようにステップ数を延長
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        ax.clear()
        ax.set_facecolor('#0a0a0a') # 視認性向上のため、わずかにグレーを混ぜた黒
        
        scans = obs
        angles = np.linspace(-2.35, 2.35, 1080)
        
        # 車両を中心とした相対座標 (進行方向を上にする)
        x_plot = scans * np.sin(-angles) # 左右
        y_plot = scans * np.cos(angles)  # 前後
        
        # 壁の点をプロット（サイズを1に下げて、遠くの壁が「面」で見えるように調整）
        ax.scatter(x_plot, y_plot, s=1, c='cyan', alpha=0.7) 
        
        # 自車をプロット
        ax.scatter(0, 0, s=100, c='red', marker='^', label='AI Car') 
        
        # --- 広角設定 ---
        ax.set_aspect('equal') # 重要：縦横比を1:1にしてコースを正確に表示
        ax.set_xlim([-15, 15]) # 左右15m
        ax.set_ylim([-5, 25])  # 前方25mまで見渡せるように拡大
        
        ax.set_title(f"AI Wide Vision - Step: {i}", color='white', fontsize=12)
        ax.axis('off')
        
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)

        if i % 100 == 0:
            print(f"ステップ {i}: 走行中...")
        if done:
            print(f"衝突しました（Step: {i}）")
            break

    plt.close(fig)

    if frames:
        print(f"{len(frames)}枚の画像から広角GIFを作成中...")
        imageio.mimsave(output_gif_path, frames, fps=30) # 滑らかに見えるようFPSを30に
        print(f"作成完了: {output_gif_path}")

if __name__ == '__main__':
    main()