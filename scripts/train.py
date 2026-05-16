import gym
import f110_gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
import os
import sys
import multiprocessing
import torch
import argparse

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.f1_env import F1TenthRL
from src import config
from src.cnn_policy import Conv1DLidarExtractor
from scripts.utils.algo_utils import get_algo_class

# ============================================================
# アルゴリズム設定
# ============================================================

# 対応アルゴリズムの一覧
SUPPORTED_ALGOS = ["ppo", "sac", "td3"]


def build_policy_kwargs(lidar_size: int, extra_size: int, frame_stack: int) -> dict:
    """Conv1D抽出器を使う場合のpolicy_kwargsを構築する。"""
    if config.USE_CNN_POLICY:
        return dict(
            features_extractor_class=Conv1DLidarExtractor,
            features_extractor_kwargs=dict(
                lidar_size=lidar_size,
                frame_stack=frame_stack,
                extra_size=extra_size,
                features_dim=256,
            ),
            net_arch=config.NET_ARCH,
        )
    else:
        return dict(net_arch=config.NET_ARCH)

def build_model_new(algo: str, env, policy_kwargs: dict):
    """新規学習用モデルを構築して返す。"""
    AlgoClass = get_algo_class(algo)

    if algo == "ppo":
        ppo_policy_kwargs = dict(**policy_kwargs, log_std_init=-1.0)
        return AlgoClass(
            "MlpPolicy",
            env,
            learning_rate=config.LEARNING_RATE,
            n_steps=config.PPO_N_STEPS,
            batch_size=config.PPO_BATCH_SIZE,
            ent_coef=config.PPO_ENT_COEF,
            policy_kwargs=ppo_policy_kwargs,
            verbose=1,
            tensorboard_log=config.LOG_DIR,
            device=config.DEVICE,
        )
    elif algo == "sac":
        return AlgoClass(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=config.TD3_BUFFER_SIZE,
            batch_size=config.TD3_BATCH_SIZE,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",
            learning_starts=config.TD3_LEARNING_STARTS,
            train_freq=1,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=config.LOG_DIR,
            device=config.DEVICE,
        )
    elif algo == "td3":
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=config.TD3_ACTION_NOISE_SIGMA * np.ones(n_actions)
        )
        return AlgoClass(
            "MlpPolicy",
            env,
            learning_rate=config.TD3_LEARNING_RATE,
            buffer_size=config.TD3_BUFFER_SIZE,
            batch_size=config.TD3_BATCH_SIZE,
            tau=0.005,
            gamma=0.99,
            action_noise=action_noise,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            learning_starts=config.TD3_LEARNING_STARTS,
            train_freq=1,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=config.LOG_DIR,
            device=config.DEVICE,
        )

def build_model_resume(algo: str, resume_path: str, env):
    """継続学習用モデルをロードして返す。"""
    AlgoClass = get_algo_class(algo)
    custom_objects = {
        "learning_rate": config.LEARNING_RATE if algo == "ppo" else 1e-4,
        "batch_size": config.PPO_BATCH_SIZE if algo == "ppo" else config.TD3_BATCH_SIZE,
    }
    return AlgoClass.load(
        resume_path,
        env=env,
        device=config.DEVICE,
        tensorboard_log=config.LOG_DIR,
        custom_objects=custom_objects,
    )

def make_env(rank: int):
    def _init():
        return F1TenthRL(config.MAP_PATH)
    return _init

def main():
    # --- CPU スレッド最適化 ---
    torch.set_num_threads(config.TORCH_NUM_THREADS)
    os.environ["OMP_NUM_THREADS"] = str(config.TORCH_NUM_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(config.TORCH_NUM_THREADS)
    print(f"[CPU] コア数: {multiprocessing.cpu_count()}, 使用スレッド数: {config.TORCH_NUM_THREADS}")

    # --- 引数パース ---
    parser = argparse.ArgumentParser(description="F1Tenth RL Training")
    parser.add_argument("--algo", type=str, default="ppo", choices=SUPPORTED_ALGOS,
                        help="使用アルゴリズム (ppo / sac / td3)")
    parser.add_argument("--steps", type=int, default=config.TOTAL_TIMESTEPS)
    parser.add_argument("--model", type=str, default=config.MODEL_PATH)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    algo = args.algo.lower()
    print(f"[ALGO]   アルゴリズム : {algo.upper()}")
    print(f"[DEVICE] デバイス     : {config.DEVICE}")
    print(f"[POLICY] USE_CNN_POLICY: {config.USE_CNN_POLICY}")

    # --- ディレクトリ作成 ---
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(config.MODEL_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 5万ステップごとに保存 (並列環境の場合は n_envs * 50000 ステップごと)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=checkpoint_dir,
        name_prefix=os.path.basename(args.model),
    )

    # --- 環境初期化 ---
    if algo == "ppo":
        # PPOの場合は並列数を自動で引き上げる (config.N_ENVS が 1 の場合のみ 8 に上書き)
        n_envs = 8 if config.N_ENVS == 1 else config.N_ENVS
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        print(f"[ENV]    SubprocVecEnv × {n_envs} (PPO用並列環境を自動構成)")
    else:
        env = DummyVecEnv([make_env(0)])
        print(f"[ENV]    DummyVecEnv × 1 ({algo.upper()}用単一環境)")

    # --- 観測空間サイズの取得 ---
    _sample_env = F1TenthRL(config.MAP_PATH)
    _lidar_size = _sample_env.lidar_size
    # lidar 以外のすべての成分を extra_size とする
    _extra_size = (
        _sample_env.residual_size +
        _sample_env.state_size +
        _sample_env.extra_size +
        _sample_env.racing_line_size +
        _sample_env.action_hist_size +
        _sample_env.residual_rl_size
    )
    _frame_stack = config.FRAME_STACK
    _sample_env.env.close()
    del _sample_env

    policy_kwargs = build_policy_kwargs(_lidar_size, _extra_size, _frame_stack)
    print(f"[CNN]    lidar={_lidar_size}, extra={_extra_size}, stack={_frame_stack}")

    # --- モデル構築 ---
    if args.resume:
        resume_path = args.resume
        if os.path.dirname(resume_path) == "":
            resume_path = os.path.join(config.MODEL_DIR, resume_path)
        if not resume_path.endswith(".zip"):
            resume_path += ".zip"
        print(f"[RESUME] {resume_path} をロード")
        model = build_model_resume(algo, resume_path, env)
    else:
        print(f"[NEW]    新規学習を開始")
        model = build_model_new(algo, env, policy_kwargs)

    # --- 保存パスの解決 ---
    save_path = args.model
    if os.path.dirname(save_path) == "":
        save_path = os.path.join(config.MODEL_DIR, save_path)
    if not save_path.endswith(".zip"):
        save_path += ".zip"

    # --- 学習開始 ---
    print(f"\n--- 学習開始: {algo.upper()} ---")
    print(f"中断(Ctrl+C)しても、その時点のモデルが {os.path.basename(save_path)} に保存されます。")
    
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=checkpoint_callback
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPT] 学習を中断しました。現在のモデルを保存します...")
    
    model.save(save_path)
    print(f"\n--- 完了: {save_path} ---")

if __name__ == "__main__":
    main()