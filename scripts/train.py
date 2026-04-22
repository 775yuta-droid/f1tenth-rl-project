import gym
import f110_gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import sys
import multiprocessing
import torch

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 共通モジュールのimport
from src.f1_env import F1TenthRL
from src import config

import argparse

def main():
    # --- CPU スレッド数の最適化 ---
    # config.TORCH_NUM_THREADS: 自動判定 or 環境変数 TORCH_NUM_THREADS で上書き可
    torch.set_num_threads(config.TORCH_NUM_THREADS)
    os.environ["OMP_NUM_THREADS"] = str(config.TORCH_NUM_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(config.TORCH_NUM_THREADS)
    print(f"[CPU] コア数: {multiprocessing.cpu_count()}, 使用スレッド数: {config.TORCH_NUM_THREADS}")
    print(f"      (変更する場合: TORCH_NUM_THREADS=<数> python3 scripts/train.py)")

    parser = argparse.ArgumentParser(description='F1Tenth PPO Training')
    parser.add_argument('--steps', type=int, default=config.TOTAL_TIMESTEPS, help='学習ステップ数')
    parser.add_argument('--model', type=str, default=config.MODEL_PATH, help='保存するモデルファイル名(拡張子なし)')
    parser.add_argument('--resume', type=str, default=None, help='継続学習元のモデルパス(拡張子なし)')
    args = parser.parse_args()

    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR, exist_ok=True)
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR, exist_ok=True)

    # チェックポイント用フォルダ
    checkpoint_dir = os.path.join(config.MODEL_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 学習の進捗に合わせて保存 (20万ステップごと)
    save_freq = 200000
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq, 
        save_path=checkpoint_dir,
        name_prefix=os.path.basename(args.model)
    )

    # --- 環境の初期化 (EXP-22: SubprocVecEnvによる8環境並列化) ---
    # SubprocVecEnvではラムダのクロージャが内包される問題を避けるためファクトリ関数を使用
    def make_env(rank):
        def _init():
            return F1TenthRL(config.MAP_PATH)
        return _init

    env = SubprocVecEnv([make_env(i) for i in range(config.N_ENVS)])
    # EXP-38: フレーム積層は環境内部（f1_env.py）で間引きを含めて処理するため、標準ラッパーは使用しない
    # env = VecFrameStack(env, n_stack=config.FRAME_STACK)

    if args.resume:
        # --- 継続学習: 既存モデルをロードして学習を再開 ---
        resume_path = args.resume if args.resume.endswith('.zip') else args.resume + '.zip'
        print(f"継続学習モード: {resume_path} をロード")
        model = PPO.load(
            resume_path,
            env=env,
            device=config.DEVICE,
            tensorboard_log=config.LOG_DIR,
            custom_objects={
                "ent_coef": config.PPO_ENT_COEF,
                "learning_rate": config.LEARNING_RATE
            }
        )
    else:
        # --- 新規学習 ---
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.LEARNING_RATE,
            ent_coef=config.PPO_ENT_COEF,
            policy_kwargs=dict(
                net_arch=config.NET_ARCH,
                log_std_init=-1.0,
            ),
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