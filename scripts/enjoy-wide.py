# import gym
# import f110_gym
# import numpy as np
# from stable_baselines3 import PPO
# import os
# import imageio
# import matplotlib.pyplot as plt

# class F1TenthRL(gym.Env):
#     def __init__(self, map_path):
#         super(F1TenthRL, self).__init__()
#         self.env = gym.make('f110_gym:f110-v0', map=map_path, num_agents=1)
#         # アクションスペースはtrain.pyと一致させる
#         self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
#         self.observation_space = gym.spaces.Box(low=0, high=30, shape=(1080,), dtype=np.float32)

#     def reset(self):
#         obs, reward, done, info = self.env.reset(np.array([[0.0, 0.0, 0.0]]))
#         return obs['scans'][0].astype(np.float32)

#     def step(self, action):
#         # 最新の学習設定（ステアリング感度0.8、速度2.5）に合わせるのが推奨ですが、
#         # ここでは元のコードの挙動をベースにしています。
#         steer = action[0] * 0.8  # クイック旋回設定
#         speed = 2.5              # 安定速度
#         obs, reward, done, info = self.env.step(np.array([[steer, speed]]))
#         return obs['scans'][0].astype(np.float32), reward, done, info

# def main():
#     print("--- AI走行テスト（広角LiDARプロット版）を開始します ---")
#     map_path = '/opt/f1tenth_gym/gym/f110_gym/envs/maps/levine'
#     output_gif_path = "../gif/run_simulation_wide.gif"

#     env = F1TenthRL(map_path)
#     model = PPO.load("../models/ppo_f1_final", device='cpu') #互換性問題の解決のためcpu指定
    
#     obs = env.reset()
#     frames = []

#     # 画面サイズを少し大きくして解像度を上げる
#     fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')

#     for i in range(2000): # 長い直線やコーナーも収まるようにステップ数を延長
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, done, info = env.step(action)
        
#         ax.clear()
#         ax.set_facecolor('#0a0a0a') # 視認性向上のため、わずかにグレーを混ぜた黒
        
#         scans = obs
#         angles = np.linspace(-2.35, 2.35, 1080)
        
#         # 車両を中心とした相対座標 (進行方向を上にする)
#         x_plot = scans * np.sin(-angles) # 左右
#         y_plot = scans * np.cos(angles)  # 前後
        
#         # 壁の点をプロット（サイズを1に下げて、遠くの壁が「面」で見えるように調整）
#         ax.scatter(x_plot, y_plot, s=1, c='cyan', alpha=0.7) 
        
#         # 自車をプロット
#         ax.scatter(0, 0, s=100, c='red', marker='^', label='AI Car') 
        
#         # --- 広角設定 ---
#         ax.set_aspect('equal') # 重要：縦横比を1:1にしてコースを正確に表示
#         ax.set_xlim([-15, 15]) # 左右15m
#         ax.set_ylim([-5, 25])  # 前方25mまで見渡せるように拡大
        
#         ax.set_title(f"AI Wide Vision - Step: {i}", color='white', fontsize=12)
#         ax.axis('off')
        
#         fig.canvas.draw()
#         frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#         frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         frames.append(frame)

#         if i % 100 == 0:
#             print(f"ステップ {i}: 走行中...")
#         if done:
#             print(f"衝突しました（Step: {i}）")
#             break

#     plt.close(fig)

#     if frames:
#         print(f"{len(frames)}枚の画像から広角GIFを作成中...")
#         imageio.mimsave("run_simulation_wide.gif", frames, fps=30)# 高フレームレートで滑らかに
#         print(f"作成完了: {output_gif_path}")

# if __name__ == '__main__':
#     main()

import gym
import f110_gym
import numpy as np
from stable_baselines3 import PPO
import os
import imageio
import matplotlib.pyplot as plt

# --- 設定 ---
MAP_PATH = '/opt/f1tenth_gym/gym/f110_gym/envs/maps/levine'
MODEL_PATH = "../models/ppo_f1_final"
OUTPUT_GIF_PATH = "../gif/run_simulation_wide.gif"
MAX_STEPS = 1500        # シミュレーションする総ステップ数
SKIP_FRAMES = 2         # 何フレームごとに画像を保存するか（2ならfps半分、メモリ節約）
FPS = 30                # GIFの再生速度

class F1TenthRL(gym.Env):
    def __init__(self, map_path):
        super(F1TenthRL, self).__init__()
        # train.pyと同じ設定で環境を作成
        self.env = gym.make('f110_gym:f110-v0', map=map_path, num_agents=1)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=30, shape=(1080,), dtype=np.float32)

    def reset(self):
        # スタート位置 (x, y, yaw)
        initial_pose = np.array([[0.0, 0.0, 0.0]])
        obs, reward, done, info = self.env.reset(initial_pose)
        return obs['scans'][0].astype(np.float32)

    def step(self, action):
        # train.py のロジックと一致させる
        steer = action[0] * 0.8 
        speed = 2.5 
        
        obs, reward, done, info = self.env.step(np.array([[steer, speed]]))
        return obs['scans'][0].astype(np.float32), reward, done, info

def main():
    print("--- AI走行可視化（完全版）を開始します ---")

    # 保存先ディレクトリがなければ作成
    output_dir = os.path.dirname(OUTPUT_GIF_PATH)
    if output_dir and not os.path.exists(output_dir):
        print(f"ディレクトリを作成します: {output_dir}")
        os.makedirs(output_dir)

    # 環境とモデルのロード
    env = F1TenthRL(MAP_PATH)

    # 互換性のため device='cpu' を指定。zipがない場合はエラーハンドリング
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"エラー: モデルファイルが見つかりません -> {MODEL_PATH}.zip")
        return

    model = PPO.load(MODEL_PATH, device='cpu')
    
    obs = env.reset()
    frames = []

    # 描画設定（黒背景でハイテク感を出す）
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0d0d0d')

    try:
        for i in range(MAX_STEPS):
            # 推論 (決定的モード)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            # 指定フレームごとに描画（高速化とメモリ節約）
            if i % SKIP_FRAMES == 0:
                ax.clear()
                ax.set_facecolor('#0d0d0d') # 濃いグレー背景

                scans = obs
                # F1TenthのLiDAR角度: -2.35rad(-135度) 〜 +2.35rad(+135度)
                angles = np.linspace(-2.35, 2.35, 1080)

                # 座標変換: 進行方向をY軸プラス(上)にする
                # 左側の障害物(角度プラス)は、X軸マイナス(左)に描画
                x_plot = scans * np.sin(angles) * -1 
                y_plot = scans * np.cos(angles)

                # 1. 壁（点群）のプロット
                ax.scatter(x_plot, y_plot, s=2, c='#00ffff', alpha=0.6, label='LiDAR')

                # 2. 自車のプロット（原点）
                ax.scatter(0, 0, s=150, c='#ff0055', marker='^', edgecolors='white', label='Car')

                # 3. 視界設定
                ax.set_aspect('equal')
                ax.set_xlim([-10, 10])  # 横幅 20m
                ax.set_ylim([-5, 20])   # 前方 20m, 後方 5m

                # 4. 情報テキスト表示
                steer_val = action[0] * 0.8
                steer_str = "RIGHT" if steer_val < 0 else "LEFT"
                if abs(steer_val) < 0.05: steer_str = "STRAIGHT"
                
                info_text = (
                    f"Step: {i}\n"
                    f"Steer: {steer_val:.3f} ({steer_str})\n"
                    f"Speed: 2.5 m/s"
                )
                ax.text(-9, 18, info_text, color='white', fontsize=12, 
                        bbox=dict(facecolor='black', alpha=0.7, edgecolor='white'))

                ax.axis('off') # 軸を消す

                # フレームをメモリに保存
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(frame)

            if i % 100 == 0:
                print(f"シミュレーション進行中... {i}/{MAX_STEPS}")

            if done:
                print(f"!! 衝突検知 !! Step: {i}")
                # 衝突時のフレームを目立たせる（赤くフラッシュ）
                # (任意実装: ここでbreakせずに数フレーム続けると衝突後の挙動も見えます)
                break

    except KeyboardInterrupt:
        print("\n処理を中断しました。ここまでのデータを保存します。")
    finally:
        plt.close(fig)

    # GIF保存
    if frames:
        print(f"GIF生成中... ({len(frames)} frames)")
        imageio.mimsave(OUTPUT_GIF_PATH, frames, fps=FPS)
        print(f"完了！ファイルを保存しました: {OUTPUT_GIF_PATH}")
    else:
        print("フレームが生成されませんでした。")

if __name__ == '__main__':
    main()