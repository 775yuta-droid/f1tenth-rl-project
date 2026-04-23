"""
CPU最適設定（TORCH_NUM_THREADS）を特定するためのベンチマークスクリプト
"""
import os
import sys
import time
import torch
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# プロジェクトルートをパスに追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.f1_env import F1TenthRL
from src import config

def make_env(rank):
    def _init():
        return F1TenthRL(config.MAP_PATH)
    return _init

def run_benchmark(num_threads, n_envs=1, total_timesteps=2000):
    # スレッド数の設定
    torch.set_num_threads(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    
    # 環境の作成
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        device="cpu"
    )
    
    print(f"計測中: Threads={num_threads}, Envs={n_envs} ...", end="", flush=True)
    
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    end_time = time.time()
    
    duration = end_time - start_time
    sps = total_timesteps / duration
    
    print(f" {sps:.2f} SPS")
    
    env.close()
    return sps

def main():
    cpu_count = multiprocessing.cpu_count()
    print(f"システムコア数: {cpu_count}")
    print("-" * 30)
    
    results = []
    
    # スレッド数の候補 (1, 2, 4, 8, 16 など、コア数に応じて)
    thread_counts = [1, 2, 4, 8]
    thread_counts = [t for t in thread_counts if t <= cpu_count]
    
    # Envs の候補 (現在は config.N_ENVS を使用、または 1, 4 など)
    env_counts = [1, config.N_ENVS] if config.N_ENVS > 1 else [1]
    
    for n_env in env_counts:
        for t in thread_counts:
            sps = run_benchmark(t, n_envs=n_env)
            results.append((n_env, t, sps))
            
    print("\n" + "="*40)
    print(f"{'Envs':>5} | {'Threads':>8} | {'SPS':>10}")
    print("-" * 40)
    
    best_sps = 0
    best_config = None
    
    for n_env, t, sps in results:
        print(f"{n_env:>5} | {t:>8} | {sps:>10.2f}")
        if sps > best_sps:
            best_sps = sps
            best_config = (n_env, t)
            
    print("="*40)
    print(f"推奨設定: TORCH_NUM_THREADS={best_config[1]} (Envs={best_config[0]})")
    print(f"実行方法: TORCH_NUM_THREADS={best_config[1]} python3 scripts/train.py")

if __name__ == "__main__":
    main()
