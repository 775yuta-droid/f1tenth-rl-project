#!/usr/bin/env python3
"""
pretrain_bc.py — Behavioral Cloning プレトレーニング
====================================================
convert_demo.py で生成した converted_*.npz を使い、
PPO / TD3 ポリシーを「模倣学習（教師あり学習）」で初期化する。

【使い方】
  # TD3 モデルで BC プレトレーニング (推奨)
  docker compose exec f1-sim-2004 python3 scripts/pretrain_bc.py --algo td3

  # PPO モデルで BC プレトレーニング
  docker compose exec f1-sim-2004 python3 scripts/pretrain_bc.py --algo ppo

  # 保存先指定
  docker compose exec f1-sim-2004 python3 scripts/pretrain_bc.py \\
      --algo td3 --model models/my_bc_model

  # デモディレクトリ指定
  docker compose exec f1-sim-2004 python3 scripts/pretrain_bc.py \\
      --demos /workspace/demos

【出力】
  models/bc_pretrained.zip   SB3 形式モデル (train.py --resume で使用可)
  logs/bc_*/                 TensorBoard ログ

【次のステップ】
  python3 scripts/train.py --algo td3 --resume models/bc_pretrained
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.f1_env import F1TenthRL
from src.cnn_policy import Conv1DLidarExtractor
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from scripts.utils.algo_utils import get_algo_class


# ─── BC ハイパーパラメータ (config.py から読み込み可能) ─────
BC_EPOCHS     = getattr(config, "BC_EPOCHS",     500)
BC_BATCH_SIZE = getattr(config, "BC_BATCH_SIZE", 256)
BC_LR         = getattr(config, "BC_LR",         3e-4)
# ────────────────────────────────────────────────────────────


def load_demos(demo_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """demos/ 以下の converted_*.npz を全て読み込んで結合する"""
    pattern = os.path.join(demo_dir, "converted_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"デモファイルが見つかりません: {pattern}\n"
            "先に convert_demo.py を実行してください。"
        )

    obs_list, act_list = [], []
    for f in files:
        data = np.load(f)
        obs_list.append(data["observations"])
        act_list.append(data["actions"])
        n = len(data["observations"])
        print(f"  読み込み: {os.path.basename(f)}  ({n} steps)")

    observations = np.concatenate(obs_list, axis=0).astype(np.float32)
    actions      = np.concatenate(act_list, axis=0).astype(np.float32)
    print(f"合計: {len(observations)} steps  "
          f"obs: {observations.shape}  actions: {actions.shape}")
    return observations, actions


def build_policy_kwargs() -> dict:
    """config.py の設定に応じた policy_kwargs を構築"""
    env_tmp = F1TenthRL(config.MAP_PATH)
    lidar_size  = env_tmp.lidar_size
    extra_size  = (env_tmp.residual_size + env_tmp.state_size +
                   env_tmp.extra_size + env_tmp.racing_line_size +
                   env_tmp.action_hist_size + env_tmp.residual_rl_size)
    frame_stack = config.FRAME_STACK
    env_tmp.env.close()

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


def build_sb3_model(algo: str, env):
    """BC 用 SB3 モデルを構築 (学習前の空のモデル)"""
    policy_kwargs = build_policy_kwargs()

    if algo == "ppo":
        ppo_kwargs = dict(**policy_kwargs, log_std_init=-1.0)
        return PPO("MlpPolicy", env,
                   learning_rate=config.LEARNING_RATE,
                   policy_kwargs=ppo_kwargs,
                   verbose=0, device=config.DEVICE)
    elif algo == "td3":
        return TD3("MlpPolicy", env,
                   learning_rate=config.TD3_LEARNING_RATE,
                   buffer_size=1000,  # BC では不要なので最小
                   policy_kwargs=policy_kwargs,
                   verbose=0, device=config.DEVICE)
    elif algo == "sac":
        return SAC("MlpPolicy", env,
                   learning_rate=3e-4,
                   buffer_size=1000,
                   policy_kwargs=policy_kwargs,
                   verbose=0, device=config.DEVICE)
    else:
        raise ValueError(f"未対応アルゴリズム: {algo}")


def get_policy_network(model, algo: str) -> nn.Module:
    """SB3 モデルからアクター (ポリシー) ネットワークだけを取り出す"""
    if algo == "ppo":
        return model.policy
    elif algo in ("td3", "sac"):
        return model.policy
    raise ValueError(f"未対応: {algo}")


def bc_train(
    policy_net: nn.Module,
    obs_tensor: torch.Tensor,
    act_tensor: torch.Tensor,
    algo: str,
    writer: SummaryWriter,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> nn.Module:
    """
    Behavioral Cloning: MSE で (obs → action) を教師あり学習

    PPO  : policy.predict(obs) で mean_action を使用
    TD3/SAC: policy.actor(obs) で action を使用
    """
    policy_net.to(device)
    policy_net.train()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = nn.MSELoss()

    dataset = TensorDataset(obs_tensor, act_tensor)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches  = 0

        for obs_b, act_b in loader:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)

            # アクション予測
            if algo == "ppo":
                dist          = policy_net.get_distribution(obs_b)
                pred_action   = dist.distribution.mean
            else:
                # TD3 / SAC actor
                pred_action = policy_net.actor(obs_b)

            loss = loss_fn(pred_action, act_b)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        writer.add_scalar("bc/loss", avg_loss, epoch)
        writer.add_scalar("bc/lr",   optimizer.param_groups[0]["lr"], epoch)

        if epoch % 50 == 0 or epoch == 1:
            marker = " ← best" if avg_loss < best_loss else ""
            print(f"  Epoch {epoch:4d}/{epochs}  loss: {avg_loss:.6f}{marker}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    policy_net.eval()
    print(f"\nBest loss: {best_loss:.6f}")
    return policy_net


def main():
    parser = argparse.ArgumentParser(description="Behavioral Cloning プレトレーニング")
    parser.add_argument("--algo", type=str, default="td3",
                        choices=["ppo", "td3", "sac"],
                        help="使用アルゴリズム (default: td3)")
    parser.add_argument("--demos", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "demos"),
                        help="変換済みデモディレクトリ")
    parser.add_argument("--model", type=str, default=None,
                        help="保存先モデルパス (未指定: config.BC_MODEL_PATH)")
    parser.add_argument("--epochs",     type=int,   default=BC_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=BC_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=BC_LR)
    args = parser.parse_args()

    save_path = args.model
    if save_path is None:
        save_path = getattr(config, "BC_MODEL_PATH",
                            os.path.join(config.MODEL_DIR, "bc_pretrained"))
    if os.path.dirname(save_path) == "":
        save_path = os.path.join(config.MODEL_DIR, save_path)
    if not save_path.endswith(".zip"):
        save_path += ".zip"

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR,   exist_ok=True)

    print("=" * 60)
    print("  Behavioral Cloning プレトレーニング")
    print(f"  アルゴリズム : {args.algo.upper()}")
    print(f"  デモディレクトリ: {args.demos}")
    print(f"  保存先: {save_path}")
    print(f"  Epochs: {args.epochs}  Batch: {args.batch_size}  LR: {args.lr}")
    print("=" * 60)

    # ─── デモデータ読み込み ──────────────────────────────────
    print("\n[1/4] デモデータ読み込み...")
    observations, actions = load_demos(args.demos)

    obs_tensor = torch.tensor(observations, dtype=torch.float32)
    act_tensor = torch.tensor(actions,      dtype=torch.float32)

    # ─── SB3 モデル構築 ──────────────────────────────────────
    print("\n[2/4] SB3 モデル構築...")
    env = DummyVecEnv([lambda: F1TenthRL(config.MAP_PATH)])
    model = build_sb3_model(args.algo, env)
    policy_net = get_policy_network(model, args.algo)
    print(f"  ポリシーパラメータ数: "
          f"{sum(p.numel() for p in policy_net.parameters()):,}")

    # ─── BC 学習 ────────────────────────────────────────────
    print(f"\n[3/4] Behavioral Cloning 学習 ({args.epochs} epochs)...")
    writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, "bc"))
    device = config.DEVICE if torch.cuda.is_available() else "cpu"

    bc_train(
        policy_net, obs_tensor, act_tensor,
        algo=args.algo,
        writer=writer,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    writer.close()

    # ─── 保存 ────────────────────────────────────────────────
    print(f"\n[4/4] モデル保存: {save_path}")
    model.save(save_path.replace(".zip", ""))
    print(f"\n✅ 完了: {save_path}")
    print("\n次のステップ（RL ファインチューニング）:")
    print(f"  python3 scripts/train.py --algo {args.algo} "
          f"--resume {os.path.basename(save_path).replace('.zip', '')}")
    env.close()


if __name__ == "__main__":
    main()
