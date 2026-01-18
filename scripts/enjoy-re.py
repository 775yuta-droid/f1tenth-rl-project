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
        steer = action[0] * 0.4
        speed = 3.5
        obs, reward, done, info = self.env.step(np.array([[steer, speed]]))
        return obs['scans'][0].astype(np.float32), reward, done, info

def main():
    map_path = '/opt/f1tenth_gym/gym/f110_gym/envs/maps/levine'
    env = F1TenthRL(map_path)
    model = PPO.load("models/ppo_f1_final.zip")
    
    obs = env.reset()
    frames = []

    print("録画中... 最初のスタイルに戻してズームを最適化しました。")

    for i in range(1200): # 少し長めに録画
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # 描画設定（最初のスタイルを継承）
        plt.figure(figsize=(6, 6))
        plt.clf()
        plt.gca().set_facecolor('black')
        
        # LiDARの点を描画 (水色)
        # 現在の車の向きに合わせて回転させる必要がない、最初の「相対座標」表示
        angles = np.linspace(-np.pi*3/4, np.pi*3/4, 1080)
        x = obs * np.cos(angles)
        y = obs * np.sin(angles)
        plt.scatter(x, y, s=2, c='cyan', alpha=0.8) # 壁
        
        # 自車を描画 (赤い三角)
        plt.scatter(0, 0, marker='^', c='red', s=150)
        
        # ズーム倍率を調整（ここが見やすさの鍵）
        plt.xlim(-8, 8) 
        plt.ylim(-2, 12) # 前方を広く見せる
        
        plt.title(f"AI Vision - Step: {i}", color='white')
        plt.axis('off') # 余計なメモリ（軸）を消してスッキリさせる
        
        # 画像に変換
        fig = plt.gcf()
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close()
        
        if done:
            print(f"終了：{i}ステップ")
            break

    imageio.mimsave("run_simulation.gif", frames, fps=30)
    print("完了！一番見やすかったスタイルで保存しました。")

if __name__ == '__main__':
    main()