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

def main():
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR, exist_ok=True)
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR, exist_ok=True)

    # チェックポイント用フォルダ
    checkpoint_dir = os.path.join(config.MODEL_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 学習の進捗に合わせて保存（5回に分けて保存するイメージ）
    save_freq = max(config.TOTAL_TIMESTEPS // 5, 1000)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq, 
        save_path=checkpoint_dir,
        name_prefix=config.MODEL_NAME
    )

    env = F1TenthRL(config.MAP_PATH)
    env = DummyVecEnv([lambda: env])

    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=config.LEARNING_RATE,
        policy_kwargs=dict(net_arch=config.NET_ARCH),
        verbose=1, 
        tensorboard_log=config.LOG_DIR,
        device=config.DEVICE
    )

    print(f"--- 学習開始: {config.MODEL_NAME} ---")
    print(f"TensorBoard ログ: {config.LOG_DIR}")
    
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=checkpoint_callback
    )
    
    model.save(config.MODEL_PATH)
    print(f"--- 完了: {config.MODEL_PATH} ---")

if __name__ == '__main__':
    main()