import config
import os
import sys

if config.DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gym
import f110_gym
import numpy as np
from stable_baselines3 import PPO
import imageio
import matplotlib.pyplot as plt

# 共通モジュールのimport
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.f1_env import F1TenthRL

def main():
    os.makedirs(os.path.dirname(config.GIF_PATH), exist_ok=True)

    env = F1TenthRL(config.MAP_PATH)
    
    # モデルの読み込みパス
    target_model = "/workspace/models/ppo_f1_custom_map_steps10000_arch2.zip"
    
    if os.path.exists(target_model):
        model = PPO.load(target_model, device=config.DEVICE)
        print(f"チェックポイントを読み込みました: {target_model}")
    else:
        print(f"エラー: モデルファイルが見つかりません: {target_model}")
        return # モデルがない場合は終了

    obs = env.reset()
    frames = []
    collisions = 0 # 衝突回数

    # --- エラー修正：axの定義をループの外（前）に移動 ---
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0d0d0d')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    print(f"--- シミュレーション録画開始 (衝突リセット継続版) ---")
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
            frame = np.array(fig.canvas.buffer_rgba())[:, :, :3]
            frames.append(frame)

        # --- 変更点：衝突しても break せず reset する ---
        if done:
            collisions += 1
            print(f"衝突！ Step: {i} (通算: {collisions}回)")
            obs = env.reset() 

    plt.close(fig)
    if len(frames) > 0:
        imageio.mimsave(config.GIF_PATH, frames, duration=50) 
        
        sim_time = len(frames) * 0.02 
        playback_time = len(frames) / 20 

        print("-" * 30)
        print(f"【シミュレーション完了】")
        print(f"■ 総フレーム数    : {len(frames)} 枚")
        print(f"■ 衝突回数        : {collisions} 回") # カウントを表示
        print(f"■ 保存先          : {config.GIF_PATH}")
        print("-" * 30)
    else:
        print("エラー: フレームが1枚も生成されませんでした。")

if __name__ == '__main__':
    main()