"""
Conv1D カスタム特徴抽出器

development-plan.md の「■ 改善案：Conv1D」に基づき実装。

LiDAR データの空間構造（壁・コーナーの連続性）を活用するため、
MLP の前段に 1次元畳み込みブロックを挿入する。

観測ベクトルの構造:
    [lidar_t0 | lidar_t1 | lidar_t2 | lidar_t3 | vehicle_state×4 | extra_feats×4]
                (FRAME_STACK 分の lidar が先頭に並ぶ)

Conv1D ブロックでは各フレームの LiDAR を (channels=FRAME_STACK, length=lidar_size)
のテンソルとして扱い、空間的局所パターンを抽出する。
"""

import torch
import torch.nn as nn
import numpy as np
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Conv1DLidarExtractor(BaseFeaturesExtractor):
    """
    Conv1D ベースの LiDAR 特徴抽出器。

    観測ベクトルから LiDAR フレームスタック部分を取り出し、
    (batch, FRAME_STACK, lidar_size) の 3D テンソルとして
    2段の 1D 畳み込みで空間特徴を抽出した後、
    残りの状態特徴（速度・ステアリング等）と結合して返す。

    Args:
        observation_space: Gym 観測空間
        lidar_size: 1フレームあたりの LiDAR 次元数 (ダウンサンプリング後)
        frame_stack: 積層フレーム数
        extra_size: 車両状態 + 追加特徴の合計次元数 (1フレーム分)
        features_dim: 最終出力次元数
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        lidar_size: int = 216,
        frame_stack: int = 4,
        extra_size: int = 4,  # vehicle_state(2) + extra_feats(2)
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim=features_dim)

        self.lidar_size = lidar_size
        self.frame_stack = frame_stack
        self.extra_size = extra_size

        # --- Conv1D ブロック ---
        # 入力: (batch, frame_stack, lidar_size)
        # → 出力: (batch, 64, L') を Flatten して全結合へ
        self.conv_block = nn.Sequential(
            # Layer 1: kernel=7, 近接壁の形状を捉える
            nn.Conv1d(in_channels=frame_stack, out_channels=32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # L/2
            # Layer 2: kernel=5, コーナー入口など中距離パターン
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # L/4
        )

        # Conv 後の lidar_size (プーリング2回で 1/4)
        conv_out_len = lidar_size // 4
        conv_out_dim = 64 * conv_out_len

        # --- Conv 出力を圧縮する全結合 ---
        self.lidar_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_dim, 128),
            nn.ReLU(),
        )

        # --- 車両状態 + 追加特徴の全結合 ---
        # extra_size × frame_stack 分の入力
        extra_total = extra_size * frame_stack
        self.state_fc = nn.Sequential(
            nn.Linear(extra_total, 64),
            nn.ReLU(),
        )

        # --- 結合後の最終全結合 ---
        combined_dim = 128 + 64
        self.out_fc = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: (batch, total_obs_size)
            total_obs_size = (lidar_size + extra_size) * frame_stack
        """
        batch = observations.shape[0]

        # ---- 1フレームあたりの次元数 ----
        per_frame = self.lidar_size + self.extra_size

        # ---- フレーム別に split ----
        # 観測は [frame_t0, frame_t1, frame_t2, frame_t3] の順に連結されている想定
        frames = observations.view(batch, self.frame_stack, per_frame)  # (B, F, per_frame)

        lidar_frames = frames[:, :, : self.lidar_size]      # (B, F, lidar_size)
        extra_frames = frames[:, :, self.lidar_size :]      # (B, F, extra_size)

        # ---- Conv1D ----
        # (B, F, L) → Conv1d expects (B, C, L): C=frame_stack として扱う
        conv_out = self.conv_block(lidar_frames)             # (B, 64, L/4)
        lidar_feat = self.lidar_fc(conv_out)                 # (B, 128)

        # ---- 車両状態 ----
        extra_flat = extra_frames.reshape(batch, -1)         # (B, extra_size * frame_stack)
        state_feat = self.state_fc(extra_flat)               # (B, 64)

        # ---- 結合 ----
        combined = torch.cat([lidar_feat, state_feat], dim=1)  # (B, 192)
        return self.out_fc(combined)                          # (B, features_dim)
