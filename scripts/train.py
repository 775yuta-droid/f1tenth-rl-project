import gym
import f110_gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import sys

# 共通モジュールのimport
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.f1_env import F1TenthRL
import config

import argparse

def main():
    parser = argparse.ArgumentParser(description='F1Tenth PPO Training')
    parser.add_argument('--steps', type=int, default=config.TOTAL_TIMESTEPS, help='学習ステップ数')
    parser.add_argument('--model', type=str, default=config.MODEL_PATH, help='保存するモデルファイル名(拡張子なし)')
    args = parser.parse_args()

    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR, exist_ok=True)
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR, exist_ok=True)

    # チェックポイント用フォルダ
    checkpoint_dir = os.path.join(config.MODEL_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 学習の進捗に合わせて保存
    save_freq = max(args.steps // 5, 1000)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq, 
        save_path=checkpoint_dir,
        name_prefix=os.path.basename(args.model)
    )

    env = F1TenthRL(config.MAP_PATH)
    env = DummyVecEnv([lambda: env])

    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=config.LEARNING_RATE,
        ent_coef=config.PPO_ENT_COEF,
        policy_kwargs=dict(net_arch=config.NET_ARCH),
        verbose=1, 
        tensorboard_log=config.LOG_DIR,
        device=config.DEVICE
    )

    print(f"--- 学習開始: {os.path.basename(args.model)} ---")
    print(f"Total Timesteps: {args.steps}")
    print(f"TensorBoard ログ: {config.LOG_DIR}")
    
    model.learn(
        total_timesteps=args.steps,
        callback=checkpoint_callback
    )
    
    model.save(args.model)
    print(f"--- 完了: {args.model} ---")


if __name__ == '__main__':
    main()