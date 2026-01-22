import config # 設定ファイルを読み込み
import os

# config.DEVICEが 'cpu' の場合のみ、GPUを隠して見えなくする
# (これを行わないと、PyTorchがGPUメモリを少し確保してしまい、VRAMを無駄食いすることがあるため)
if config.DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gym
import f110_gym
import numpy as np
from stable_baselines3 import PPO
import imageio
import matplotlib.pyplot as plt

class F1TenthRL(gym.Env):
    def __init__(self, map_path):
        super(F1TenthRL, self).__init__()
        self.env = gym.make('f110_gym:f110-v0', map=map_path, num_agents=1)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=30, shape=(1080,), dtype=np.float32)

    def reset(self):
        obs, _, _, _ = self.env.reset(np.array([[0.0, 0.0, 0.0]]))
        return obs['scans'][0].astype(np.float32)

    def step(self, action):
        steer = action[0] * config.STEER_SENSITIVITY
        speed = config.MAX_SPEED
        obs, reward, done, info = self.env.step(np.array([[steer, speed]]))
        return obs['scans'][0].astype(np.float32), reward, done, info

def main():
    os.makedirs(os.path.dirname(config.GIF_PATH), exist_ok=True)

    env = F1TenthRL(config.MAP_PATH)
    model = PPO.load(config.MODEL_PATH, device='config.DEVICE')
    
    obs = env.reset()
    frames = []

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0d0d0d')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    print("--- シミュレーション録画開始 ---")
    for i in range(1500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if i % 2 == 0:
            ax.clear()
            ax.set_facecolor('#0a0a0c')
            scans = obs
            angles = np.linspace(-2.35, 2.35, 1080)
            x_plot = scans * np.sin(angles) * -1 
            y_plot = scans * np.cos(angles)

            ax.scatter(x_plot, y_plot, s=1.5, c='#00ffff', alpha=0.5)
            ax.scatter(0, 0, s=200, c='#ff0055', marker='^', edgecolors='white')
            
            ax.set_aspect('equal')
            ax.set_xlim([-12, 12])
            ax.set_ylim([-3, 22])
            ax.axis('off')

            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)

        if done:
            print(f"衝突！ Step: {i}")
            break

    plt.close(fig)
    imageio.mimsave(config.GIF_PATH, frames, fps=30)
    print(f"--- GIF作成完了: {config.GIF_PATH} ---")

if __name__ == '__main__':
    main()