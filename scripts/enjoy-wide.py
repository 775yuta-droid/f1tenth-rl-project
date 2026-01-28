import config
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from PIL import Image
import imageio
import argparse

# 共通モジュールのimport
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.f1_env import F1TenthRL

class MapRenderer:
    def __init__(self, map_path, fig_size=8):
        # マップメタデータの読み込み
        map_yaml_path = map_path + ".yaml"
        with open(map_yaml_path, 'r') as f:
            map_conf = yaml.safe_load(f)
        
        self.origin = map_conf['origin'] # [x, y, theta]
        self.resolution = map_conf['resolution']
        img_name = map_conf['image']
        
        map_dir = os.path.dirname(map_path)
        img_path = os.path.join(map_dir, img_name)
        
        # 画像読み込み
        img = Image.open(img_path)
        self.map_img = np.array(img)
        self.height, self.width = self.map_img.shape
        
        # グラフ設定
        aspect = self.width / self.height
        self.fig, self.ax = plt.subplots(figsize=(fig_size * aspect, fig_size), facecolor='#121212')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # 初期描画（背景）
        self.ax.imshow(self.map_img, cmap='gray', origin='upper')
        self.ax.axis('off')

        # 描画オブジェクトの初期化
        self.trail, = self.ax.plot([], [], color='#00aaff', alpha=0.5, linewidth=1, label='Trail')
        self.scans_scatter = self.ax.scatter([], [], s=1, c='#00ffff', alpha=0.3)
        self.car_dot, = self.ax.plot([], [], 'o', color='#ff0055', markersize=8, markeredgecolor='white', zorder=5)
        self.car_arrow = None # 後で作成
        
        self.hud_text = self.ax.text(20, 40, '', color='#00ff00', fontsize=12, fontfamily='monospace',
                                    bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
        
        # 軌跡データ
        self.trail_x = []
        self.trail_y = []

    def world_to_pixel(self, x, y):
        px = (x - self.origin[0]) / self.resolution
        py = self.height - (y - self.origin[1]) / self.resolution
        return px, py

    def update(self, car_state, scans, action, reward, step, collisions):
        car_x, car_y, car_theta, car_vel = car_state
        px, py = self.world_to_pixel(car_x, car_y)

        # 軌跡の更新
        self.trail_x.append(px)
        self.trail_y.append(py)
        self.trail.set_data(self.trail_x, self.trail_y)

        # 自車の位置
        self.car_dot.set_data([px], [py])

        # 向きの矢印（Arrowは更新が少し特殊なので毎回消して描くか、Patchを使う）
        if self.car_arrow:
            self.car_arrow.remove()
        
        arrow_len = 15
        dx = arrow_len * np.cos(car_theta)
        dy = -arrow_len * np.sin(car_theta) # 画像座標系
        self.car_arrow = self.ax.arrow(px, py, dx, dy, head_width=6, head_length=8, fc='#ff0055', ec='white', zorder=6)

        # LiDAR点群
        angles = np.linspace(-2.35, 2.35, 1080) + car_theta
        scan_x_world = car_x + scans * np.cos(angles)
        scan_y_world = car_y + scans * np.sin(angles)
        scan_px, scan_py = self.world_to_pixel(scan_x_world, scan_y_world)
        self.scans_scatter.set_offsets(np.c_[scan_px, scan_py])

        # HUD
        info_str = (
            f"STEP: {step:04d}\n"
            f"SPD : {car_vel:.2f} m/s\n"
            f"STR : {action[0]:.2f}\n"
            f"ACC : {action[1]:.2f}\n"
            f"RWD : {reward:.2f}\n"
            f"COL : {collisions}"
        )
        self.hud_text.set_text(info_str)

        # 画面キャプチャ
        self.fig.canvas.draw()
        frame = np.array(self.fig.canvas.buffer_rgba())[:, :, :3]
        return frame

def main():
    parser = argparse.ArgumentParser(description='F1Tenth PPO Model Viewer')
    parser.add_argument('--steps', type=int, default=1500, help='最大シミュレーションステップ数')
    parser.add_argument('--model', type=str, default=None, help='モデルファイルのパス(拡張子なし)')
    parser.add_argument('--save', type=str, default=config.GIF_PATH, help='保存先のパス')
    parser.add_argument('--no-render', action='store_true', help='GIFを生成しない(デバッグ用)')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    # 環境の初期化
    env = F1TenthRL(config.MAP_PATH)
    
    # モデルの読み込み
    target_model = args.model if args.model else config.MODEL_PATH
    if not target_model.endswith(".zip"):
        target_model += ".zip"
    
    if os.path.exists(target_model):
        model = PPO.load(target_model, device=config.DEVICE)
        print(f"モデルをロードしました: {target_model}")
    else:
        print(f"エラー: モデルファイルが見つかりません: {target_model}")
        return

    # 描画クラスの初期化
    renderer = MapRenderer(config.MAP_PATH)

    obs = env.reset()
    frames = []
    collisions = 0
    total_reward = 0
    
    print(f"--- シミュレーション開始 (最大 {args.steps} ステップ) ---")
    
    try:
        for i in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # 車両状態の取得
            try:
                state = env.env.sim.agents[0].state
                car_state = (state[0], state[1], state[2], state[3]) # x, y, theta, velocity
            except:
                car_state = (0, 0, 0, 0)

            # 描画更新 (2ステップに1回)
            if i % 2 == 0 and not args.no_render:
                frame = renderer.update(car_state, obs, action, reward, i, collisions)
                frames.append(frame)
                
                if (i // 2) % 50 == 0:
                    print(f"レンダリング中... Step: {i}")

            if done:
                collisions += 1
                print(f"衝突！ Step: {i} (累積: {collisions}, 報酬累計: {total_reward:.1f})")
                obs = env.reset()
                total_reward = 0
                # 衝突時に軌跡をリセットするかどうかは好みだが、ここでは継続して描画する

    except KeyboardInterrupt:
        print("\n中断されました。")
    finally:
        plt.close(renderer.fig)
        
        if len(frames) > 0 and not args.no_render:
            print(f"GIF生成中... ({len(frames)} frames)")
            imageio.mimsave(args.save, frames, duration=40)
            print(f"保存完了: {args.save}")
        else:
            print("保存はスキップされました。")

if __name__ == '__main__':
    main()