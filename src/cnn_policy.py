"""
Conv1D カスタム特徴抽出器

development-plan.md の「■ 改善案：Conv1D」に基づき実装。

LiDAR データの空間構造（壁・コーナーの連続性）を活用するため、
MLP の前段に 1次元畳み込みブロックを挿入する。

【観測ベクトルの実際のレイアウト (f1_env.py が生成する順序)】
    frame 0 (最新) : [lidar(lidar_size) | vehicle_state(extra_size)]
    frame 1 (1つ前): [lidar(lidar_size) | vehicle_state(extra_size)]
    frame 2 (2つ前): [lidar(lidar_size) | vehicle_state(extra_size)]
    frame 3 (最古) : [lidar(lidar_size) | vehicle_state(extra_size)]

obs_buffer は最新フレームが先頭 (逆順) で stacked_obs に連結される。
forward() では flip して「最古→最新」の正しい時間順序に直してから Conv1D へ渡す。
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
        extra_size: int = 5,  # 2(state) + 3(extra_feats) = 5 に変更 (EXP-49)
        features_dim: int = 256, # 512 -> 256 に戻す（拡大→縮小の無駄を除去）
    ):
        super().__init__(observation_space, features_dim=features_dim)

        self.lidar_size = lidar_size
        self.frame_stack = frame_stack
        self.extra_size = extra_size

        # --- Conv1D ブロック ---
        # 入力: (batch, frame_stack, lidar_size)
        self.conv_block = nn.Sequential(
            # Layer 1: kernel=7, 受容野 8.75° (7/216*270)
            nn.Conv1d(in_channels=frame_stack, out_channels=32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # L/2
            # Layer 2: kernel=9, padding=4 (コーナー認識向上)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # L/4
            # Layer 3: kernel=5, padding=2
            # ※ EXP-46: Poolingなしで情報を保持しつつ受容野を拡大
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(16)  # (B, 128, L/4) → (B, 128, 16)

        # Conv 後の lidar_size
        conv_out_len = 16
        conv_out_dim = 128 * conv_out_len # 2048

        # --- Conv 出力を圧縮する全結合 ---
        self.lidar_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_dim, 256), # 128 -> 256
            nn.ReLU(),
        )

        # --- 車両状態 + 追加特徴の全結合 ---
        # extra_size × frame_stack 分の入力
        extra_total = extra_size * frame_stack
        self.state_fc = nn.Sequential(
            nn.Linear(extra_total, 32),
            nn.ReLU(),
        )

        # --- 結合後の最終全結合 ---
        combined_dim = 256 + 32
        self.out_fc = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: (batch, total_obs_size)
            total_obs_size = (lidar_size + extra_size) * frame_stack

        f1_env.py の obs_buffer は「最新フレームが先頭」の逆順で連結される。
        Conv1D が時間方向のパターン（壁の接近など）を正しく学習できるよう、
        ここで flip して「最古フレーム → 最新フレーム」の順に並び替える。
        """
        batch = observations.shape[0]

        # ---- 1フレームあたりの次元数 ----
        per_frame = self.lidar_size + self.extra_size

        # ---- フレーム別に split ----
        # 観測: [frame_newest | frame_1ago | frame_2ago | frame_oldest]
        # → (B, frame_stack, per_frame) に reshape
        frames = observations.view(batch, self.frame_stack, per_frame)  # (B, F, per_frame)

        # 時間軸を反転: [newest, 1ago, 2ago, oldest] → [oldest, 2ago, 1ago, newest]
        # これにより Conv1D の「チャンネル軸=時間軸」が正しい順序になる
        frames = torch.flip(frames, dims=[1])                           # (B, F, per_frame)

        lidar_frames = frames[:, :, : self.lidar_size]      # (B, F, lidar_size)
        extra_frames = frames[:, :, self.lidar_size :]      # (B, F, extra_size)

        # ---- Conv1D ----
        # (B, F, L) → Conv1d expects (B, C, L): C=frame_stack (時間チャンネル) として扱う
        conv_out = self.conv_block(lidar_frames)             # (B, 128, 54)
        conv_out = self.pool(conv_out)                       # (B, 128, 16)
        lidar_feat = self.lidar_fc(conv_out)                 # (B, 256)

        # ---- 車両状態 ----
        extra_flat = extra_frames.reshape(batch, -1)         # (B, extra_size * frame_stack)
        state_feat = self.state_fc(extra_flat)               # (B, 32)

        # ---- 結合 ----
        combined = torch.cat([lidar_feat, state_feat], dim=1)  # (B, 192)
        return self.out_fc(combined)                          # (B, features_dim)
