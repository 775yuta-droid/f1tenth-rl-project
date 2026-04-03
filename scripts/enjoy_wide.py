import os
import sys

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from PIL import Image
import imageio
import argparse
from src import config
from src.f1_env import F1TenthRL
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

class MapRenderer:
    def __init__(self, map_path, car_params={'length': 0.465, 'width': 0.19}, fig_size=8):
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
        
        # 車両の外形（ポリゴン）
        self.car_len = car_params['length']
        self.car_width = car_params['width']
        self.car_polygon = plt.Polygon([(0,0), (0,0), (0,0), (0,0)], closed=True, 
                                      fill=True, color='#ff0055', alpha=0.6, zorder=4)
        self.ax.add_patch(self.car_polygon)
        
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

        # 車両の外形（ポリゴン）の更新
        # シミュレーターの基準点 (px, py) は後輪軸の中心。
        # 車両の中心 (CG付近) はそこから前方（theta方向）にシフトしている。
        # 0.465m長、0.33mホイールベース、対称オーバーハングの場合、オフセットは約 0.165m
        offset_dist = 0.165 
        off_x = (offset_dist / self.resolution) * np.cos(car_theta)
        off_y = -(offset_dist / self.resolution) * np.sin(car_theta) # y軸反転
        
        center_px = px + off_x
        center_py = py + off_y

        l_px = self.car_len / self.resolution
        w_px = self.car_width / self.resolution
        
        # 頂点計算
        cos_t = np.cos(car_theta)
        sin_t = np.sin(car_theta)
        # 4隅の相対座標 (車両中心基準)
        corners = np.array([
            [l_px/2, w_px/2],   # 前左
            [l_px/2, -w_px/2],  # 前右
            [-l_px/2, -w_px/2], # 後右
            [-l_px/2, w_px/2]   # 後左
        ])
        # 回転
        rotated_corners = []
        for cx, cy in corners:
            rx = cx * cos_t - cy * sin_t
            ry = cx * sin_t + cy * cos_t
            rotated_corners.append([center_px + rx, center_py - ry])
        
        self.car_polygon.set_xy(rotated_corners)

        # 向きの矢印
        if self.car_arrow:
            self.car_arrow.remove()
        
        # マップ解像度(0.075)に合わせて矢印のサイズを調整
        arrow_len = 10 
        dx = arrow_len * np.cos(car_theta)
        dy = -arrow_len * np.sin(car_theta) # 画像座標系(y軸反転)
        self.car_arrow = self.ax.arrow(px, py, dx, dy, head_width=4, head_length=5, fc='#ff0055', ec='white', zorder=6)

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

    # 保存先の調整 (ディレクトリ指定がない場合は config.GIF_DIR を使用)
    save_path = args.save
    if os.path.dirname(save_path) == '':
        save_path = os.path.join(config.GIF_DIR, save_path)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 環境の初期化
    env_single = F1TenthRL(config.MAP_PATH)
    env = DummyVecEnv([lambda: env_single])
    # EXP_21: フレーム積層の適用
    env = VecFrameStack(env, n_stack=config.FRAME_STACK)
    
    # モデルの読み込み (ディレクトリ指定がない場合は config.MODEL_DIR を指定)
    if args.model:
        if os.path.dirname(args.model) == '':
            target_model = os.path.join(config.MODEL_DIR, args.model)
        else:
            target_model = args.model
    else:
        target_model = config.MODEL_PATH
    if not target_model.endswith(".zip"):
        target_model += ".zip"
    
    if os.path.exists(target_model):
        model = PPO.load(target_model, device=config.DEVICE)
        print(f"モデルをロードしました: {target_model}")
    else:
        print(f"エラー: モデルファイルが見つかりません: {target_model}")
        return

    # 描画クラスの初期化 (ユーザー希望の 0.465 x 0.19 を強制)
    car_params = {'length': 0.465, 'width': 0.19}
    renderer = MapRenderer(config.MAP_PATH, car_params=car_params)

    obs = env.reset()
    frames = []
    collisions = 0
    total_reward = 0
    
    print(f"--- シミュレーション開始 (最大 {args.steps} ステップ) ---")
    
    try:
        for i in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            
            # VecEnv (VecFrameStack) の戻り値は常に要素1のリスト
            reward = rewards[0]
            done   = dones[0]
            info   = infos[0]
            
            raw_scan = info.get('raw_scan', np.zeros(1080))
            total_reward += reward

            # 車両状態の取得
            try:
                # VecEnv なので env.envs[0] で元のインスタンスにアクセス
                state = env_single.env.sim.agents[0].state
                # state[0]: x, state[1]: y, state[4]: yaw (psi), state[3]: velocity
                # state[2] はステアリング角なので、向きには state[4] を使うのが正しい
                car_state = (state[0], state[1], state[4], state[3]) 
            except Exception as e:
                car_state = (0, 0, 0, 0)

            # 描画更新 (2ステップに1回)
            if i % 2 == 0 and not args.no_render:
                # action[0] (現在のエージェントの操作) を渡す
                frame = renderer.update(car_state, raw_scan, action[0], reward, i, collisions)
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
            print(f"動画生成中... ({len(frames)} frames)")
            if save_path.lower().endswith('.mp4'):
                # MP4の場合 (fps=25 は duration=40ms に相当)
                imageio.mimsave(save_path, frames, fps=25, quality=8, macro_block_size=16)
            else:
                # GIFの場合
                imageio.mimsave(save_path, frames, duration=40)
            print(f"保存完了: {save_path}")
        else:
            print("保存はスキップされました。")

if __name__ == '__main__':
    main()