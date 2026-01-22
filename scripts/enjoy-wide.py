# import config # 設定ファイルを読み込み
# import os

# # config.DEVICEが 'cpu' の場合のみ、GPUを隠して見えなくする
# # (これを行わないと、PyTorchがGPUメモリを少し確保してしまい、VRAMを無駄食いすることがあるため)
# if config.DEVICE == "cpu":
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import gym
# #import gymnasium as gym
# import f110_gym
# import numpy as np
# from stable_baselines3 import PPO
# import imageio
# import matplotlib.pyplot as plt

# class F1TenthRL(gym.Env):
#     def __init__(self, map_path):
#         super(F1TenthRL, self).__init__()
#         self.env = gym.make('f110_gym:f110-v0', map=map_path, num_agents=1)
#         self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
#         self.observation_space = gym.spaces.Box(low=0, high=30, shape=(1080,), dtype=np.float32)

#     def reset(self):
#         obs, _, _, _ = self.env.reset(np.array([[0.0, 0.0, 0.0]]))
#         return obs['scans'][0].astype(np.float32)

#     def step(self, action):
#         steer = action[0] * config.STEER_SENSITIVITY
#         speed = config.MAX_SPEED
#         obs, reward, done, info = self.env.step(np.array([[steer, speed]]))
#         return obs['scans'][0].astype(np.float32), reward, done, info

# def main():
#     os.makedirs(os.path.dirname(config.GIF_PATH), exist_ok=True)

#     env = F1TenthRL(config.MAP_PATH)
#     model = PPO.load(config.MODEL_PATH, device='config.DEVICE')
    
#     obs = env.reset()
#     frames = []

#     fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0d0d0d')
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

#     print("--- シミュレーション録画開始 ---")
#     for i in range(1500):
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, info = env.step(action)

#         if i % 2 == 0:
#             ax.clear()
#             ax.set_facecolor('#0a0a0c')
#             scans = obs
#             angles = np.linspace(-2.35, 2.35, 1080)
#             x_plot = scans * np.sin(angles) * -1 
#             y_plot = scans * np.cos(angles)

#             ax.scatter(x_plot, y_plot, s=1.5, c='#00ffff', alpha=0.5)
#             ax.scatter(0, 0, s=200, c='#ff0055', marker='^', edgecolors='white')
            
#             ax.set_aspect('equal')
#             ax.set_xlim([-12, 12])
#             ax.set_ylim([-3, 22])
#             ax.axis('off')

#             fig.canvas.draw()
#             frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#             frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#             frames.append(frame)

#         if done:
#             print(f"衝突！ Step: {i}")
#             break

#     plt.close(fig)
#     imageio.mimsave(config.GIF_PATH, frames, fps=30)
#     print(f"--- GIF作成完了: {config.GIF_PATH} ---")

# if __name__ == '__main__':
#     main()

import config
import os
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
        self.env = gym.make('f110-v0', map=map_path, num_agents=1)
        # 2次元に修正 (train.pyと一致させる)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=30, shape=(1080,), dtype=np.float32)

    def reset(self):
        initial_poses = np.array([[0.0, 0.0, 0.0]])
        result = self.env.reset(poses=initial_poses)
        if isinstance(result, tuple):
            obs = result[0]
        else:
            obs = result
        return obs['scans'][0].astype(np.float32)

    def step(self, action):
        steer = action[0] * config.STEER_SENSITIVITY
        # 速度の計算ロジックを train.py と合わせる
        speed = config.MIN_SPEED + (action[1] + 1.0) * (config.MAX_SPEED - config.MIN_SPEED) / 2.0
        obs, reward, done, info = self.env.step(np.array([[steer, speed]]))
        return obs['scans'][0].astype(np.float32), reward, done, info

def main():
    os.makedirs(os.path.dirname(config.GIF_PATH), exist_ok=True)

    env = F1TenthRL(config.MAP_PATH)
    # device='config.DEVICE' ではなく config.DEVICE に修正
    model = PPO.load(config.MODEL_PATH, device=config.DEVICE)
    
    obs = env.reset()
    frames = []

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0d0d0d')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    print(f"--- シミュレーション録画開始 (Action 2D 対応) ---")
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
            # 現在の速度を可視化テキストとして追加するのもアリ
            ax.scatter(0, 0, s=200, c='#ff0055', marker='^', edgecolors='white')
            
            ax.set_aspect('equal')
            ax.set_xlim([-12, 12])
            ax.set_ylim([-3, 22])
            ax.axis('off')

            fig.canvas.draw()
            fig.canvas.draw()
            frame = np.array(fig.canvas.buffer_rgba())[:, :, :3] # RGBAからRGBのみ取得
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)

        if done:
            print(f"衝突または終了 Step: {i}")
            break

    plt.close(fig)
    if len(frames) > 0:
        # 1000ms / 20fps = 50ms per frame
        imageio.mimsave(config.GIF_PATH, frames, duration=50) 
        
        # 1フレーム = 0.01秒(1step) × 2 (2回に1回保存) = 0.02秒
        sim_time = len(frames) * 0.02 
        playback_time = len(frames) / 20  # 20fps設定の場合

        print("-" * 30)
        print(f"【シミュレーション完了】")
        print(f"■ 総フレーム数    : {len(frames)} 枚")
        print(f"■ 物理走行時間    : {sim_time:.2f} 秒")
        print(f"■ 動画の再生時間  : {playback_time:.2f} 秒")
        print(f"■ 再生スピード    : {sim_time / playback_time:.2f} 倍速 (1.0で等倍)")
        print(f"■ 保存先          : {config.GIF_PATH}")
        print("-" * 30)
    else:
        print("エラー: フレームが1枚も生成されませんでした。")

if __name__ == '__main__':
    main()