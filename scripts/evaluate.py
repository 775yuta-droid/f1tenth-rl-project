import config
import os
import sys
import numpy as np
from stable_baselines3 import PPO
import argparse
import time
import csv
import json
import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.f1_env import F1TenthRL


def main():
    parser = argparse.ArgumentParser(description='F1Tenth Model Benchmark Evaluator')
    parser.add_argument('--episodes', type=int, default=10, help='評価するエピソード数')
    parser.add_argument('--max_steps', type=int, default=2000, help='1エピソードあたりの最大ステップ数')
    parser.add_argument('--model', type=str, default=None, help='モデルファイルのパス(拡張子なし)')
    args = parser.parse_args()

    # 環境の初期化
    env = F1TenthRL(config.MAP_PATH)
    print(f"現在の観測空間の形状: {env.observation_space.shape}")

    # モデルの読み込み
    target_model = args.model if args.model else config.MODEL_PATH
    if not target_model.endswith('.zip'):
        target_model += '.zip'

    if os.path.exists(target_model):
        try:
            model = PPO.load(target_model, device=config.DEVICE)
            print(f"モデルをロードしました: {target_model}")
        except ValueError as e:
            print("--- 読み込みエラー ---")
            print(f"モデル '{target_model}' の読み込みに失敗しました。")
            print("観測空間の次元設定（LIDAR_DOWNSAMPLE_FACTOR 等）が学習時と異なっている可能性があります。")
            print(f"詳細: {e}")
            return
    else:
        print(f"エラー: モデルファイルが見つかりません: {target_model}")
        return

    print(f"\n--- ベンチマーク開始 ({args.episodes} エピソード) ---")

    results = {
        "steps": [],
        "rewards": [],
        "avg_speeds": [],
        "collisions": 0,
        "success": 0
    }
    statuses = []

    start_time = time.time()

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        speeds = []

        while not done and ep_steps < args.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            try:
                speed = env.env.sim.agents[0].state[3]
                speeds.append(speed)
            except:
                pass

            ep_reward += reward
            ep_steps += 1

        results["steps"].append(ep_steps)
        results["rewards"].append(ep_reward)
        results["avg_speeds"].append(np.mean(speeds) if speeds else 0)

        # 成功/衝突の判定
        # F1Tenth gym では done=True が衝突（壁接触によるエピソード終了）を意味する
        # done=False のままループを抜けた場合は最大ステップ数到達（完走）
        if done:
            results["collisions"] += 1
            status = "Collision"
        else:
            results["success"] += 1
            status = "Success (Max Steps)"

        statuses.append(status)
        print(f"Episode {ep+1:02d}: Steps={ep_steps:4d}, Reward={ep_reward:7.1f}, Speed={np.mean(speeds):.2f}m/s, {status}")

    total_time = time.time() - start_time

    print("\n" + "="*40)
    print("📊 最終ベンチマーク結果")
    print("="*40)
    print(f"モデル: {os.path.basename(target_model)}")
    print(f"総計時間: {total_time:.2f} 秒")
    print(f"成功率 (完走): {results['success'] / args.episodes * 100:.1f}%")
    print(f"衝突率: {results['collisions'] / args.episodes * 100:.1f}%")
    print("-"*40)
    print(f"平均ステップ数: {np.mean(results['steps']):.1f} steps")
    print(f"平均累積報酬: {np.mean(results['rewards']):.1f}")
    print(f"全体平均速度: {np.mean(results['avg_speeds']):.2f} m/s")
    print(f"最高平均速度: {np.max(results['avg_speeds']):.2f} m/s")
    print("="*40)

    # 結果を CSV と JSON に保存
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_path  = os.path.join(log_dir, f"benchmark_{timestamp}.csv")
    json_path = os.path.join(log_dir, f"benchmark_{timestamp}.json")

    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["episode", "steps", "reward", "avg_speed", "status"])
        for i in range(args.episodes):
            writer.writerow([i+1, results["steps"][i], results["rewards"][i], results["avg_speeds"][i], statuses[i]])

    with open(json_path, "w") as jsonfile:
        json.dump({
            "model": os.path.basename(target_model),
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "total_time_sec": total_time,
            "success_rate": results['success'] / args.episodes,
            "collision_rate": results['collisions'] / args.episodes,
            "avg_steps": float(np.mean(results['steps'])),
            "avg_reward": float(np.mean(results['rewards'])),
            "avg_speed": float(np.mean(results['avg_speeds'])),
            "per_episode": [
                {"episode": i+1, "steps": results["steps"][i],
                 "reward": results["rewards"][i],
                 "avg_speed": results["avg_speeds"][i],
                 "status": statuses[i]}
                for i in range(args.episodes)
            ]
        }, jsonfile, indent=2, ensure_ascii=False)

    print(f"結果を保存しました: {csv_path}")
    print(f"            と: {json_path}")


if __name__ == '__main__':
    main()
