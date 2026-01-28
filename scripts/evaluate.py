import config
import os
import sys
import numpy as np
from stable_baselines3 import PPO
import argparse
import time

# 共通モジュールのimport
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

    print(f"\n--- ベンチマーク開始 ({args.episodes} エピソード) ---")
    
    results = {
        "steps": [],
        "rewards": [],
        "avg_speeds": [],
        "collisions": 0,
        "success": 0
    }

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
            
            # 速度の取得
            try:
                speed = env.env.sim.agents[0].state[3]
                speeds.append(speed)
            except:
                pass
                
            ep_reward += reward
            ep_steps += 1
        
        # 記録
        results["steps"].append(ep_steps)
        results["rewards"].append(ep_reward)
        results["avg_speeds"].append(np.mean(speeds) if speeds else 0)
        
        if done:
            results["collisions"] += 1
            status = "Collision"
        else:
            results["success"] += 1
            status = "Success (Max Steps)"
            
        print(f"Episode {ep+1:02d}: Steps={ep_steps:4d}, Reward={ep_reward:7.1f}, Speed={np.mean(speeds):.2f}m/s, {status}")

    total_time = time.time() - start_time
    
    # 集計結果の表示
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

if __name__ == '__main__':
    main()
