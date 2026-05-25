#!/usr/bin/env python3
"""
convert_demo.py — 実機デモ → シミュレータ互換フォーマット変換
============================================================
実機 F1Tenth で録画した demo_*.npz を、このリポジトリの RL 訓練用
(observations, actions) ペアに変換する。

【使い方】
  # 単一ファイル変換
  docker compose exec f1-sim-2004 python3 scripts/convert_demo.py \\
      --input demos/demo_20260525_101000.npz

  # demos/ ディレクトリ内の全ファイルを一括変換
  docker compose exec f1-sim-2004 python3 scripts/convert_demo.py --all

【変換ロジック】
  LiDAR:
    1. 実機 Hokuyo (1080 beams, 270°) → f1_env._get_obs() と同じ前処理
    2. ダウンサンプリング (min-pool, LIDAR_DOWNSAMPLE_FACTOR)
    3. 反転正規化: 1.0 - clip(x, 0, 30) / 30
  Action:
    steer_norm = clip(steer / REAL_STEER_MAX, -1, 1)
    speed_norm = clip((speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED) * 2 - 1, -1, 1)
  追加特徴 (vehicle_state / extra_features / action_history):
    前ステップのアクションを用いて疑似生成

【出力フォーマット】
  observations : (N, obs_dim * FRAME_STACK)  float32
  actions      : (N, 2)                       float32
"""

import os
import sys
import glob
import argparse
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config


# ─── 設定 ──────────────────────────────────────────────────
REAL_STEER_MAX = 0.4189   # Hokuyo / Futaba 実機最大ステア角 [rad]
LIDAR_BEAMS    = 1080     # 実機 LiDAR ビーム数 (270° = 1080 本)
# ────────────────────────────────────────────────────────────


def preprocess_lidar(raw: np.ndarray) -> np.ndarray:
    """
    実機 LiDAR 1 フレーム (1080,) をシム互換 obs ベクトルに変換。

    Returns
    -------
    obs_single : (lidar_size + extra_size,) float32
        FRAME_STACK 前の 1 フレーム分の観測ベクトル
        (FRAME_STACK は呼び出し側で積層する)
    """
    # 1. クリッピング & NaN 除去
    scans = np.clip(
        np.nan_to_num(raw, nan=config.LIDAR_MAX_RANGE,
                      posinf=config.LIDAR_MAX_RANGE, neginf=0.0),
        0.0, config.LIDAR_MAX_RANGE
    ).astype(np.float32)

    # 実機は 270° 1080 beam → そのまま使用
    # シムは中央 1080 点をスライスしているので互換
    if len(scans) != LIDAR_BEAMS:
        # ビーム数が違う場合は線形補間でリサンプル
        indices = np.linspace(0, len(scans) - 1, LIDAR_BEAMS)
        scans = np.interp(indices, np.arange(len(scans)), scans).astype(np.float32)

    # 2. ダウンサンプリング (min-pool)
    lidar_size = LIDAR_BEAMS // config.LIDAR_DOWNSAMPLE_FACTOR
    downsampled = scans.reshape(lidar_size, config.LIDAR_DOWNSAMPLE_FACTOR).min(axis=1)

    # 3. 反転正規化 (近い壁 → 1.0, 遠い空間 → 0.0)
    lidar_norm = 1.0 - downsampled / config.LIDAR_MAX_RANGE

    parts = [lidar_norm]

    # 4. 追加特徴（実機では車両状態が不明なため 0 埋め）
    if config.INCLUDE_LIDAR_RESIDUAL:
        parts.append(np.zeros(lidar_size, dtype=np.float32))

    if config.INCLUDE_VEHICLE_STATE:
        # vel と steer は action から後で上書き
        parts.append(np.zeros(2, dtype=np.float32))

    if config.INCLUDE_EXTRA_FEATURES:
        # front_dist, min_dist, lr_asymmetry を LiDAR から計算
        _H = downsampled
        _front = _H[lidar_size // 2 - 32 : lidar_size // 2 + 32]
        front_raw = float(np.min(_front)) if len(_front) > 0 else config.LIDAR_MAX_RANGE
        min_raw   = float(np.min(_H))

        front_feat = 1.0 - front_raw / config.LIDAR_MAX_RANGE
        min_feat   = 1.0 - min_raw   / config.LIDAR_MAX_RANGE

        _left_diag  = np.min(_H[lidar_size // 2 + 13 : lidar_size // 2 + 38])
        _right_diag = np.min(_H[lidar_size // 2 - 38 : lidar_size // 2 - 13])
        lr_asym = (_left_diag - _right_diag) / config.LIDAR_MAX_RANGE

        parts.append(np.array([front_feat, min_feat, lr_asym], dtype=np.float32))

    if config.INCLUDE_RACING_LINE:
        # レーシングライン特徴は実機データでは取得不可 → 0 埋め
        parts.append(np.zeros(4, dtype=np.float32))

    if config.INCLUDE_ACTION_HISTORY:
        # 前ステップ行動: 後でループ内で上書きする
        parts.append(np.zeros(2, dtype=np.float32))

    return np.concatenate(parts).astype(np.float32)


def normalize_action(steer_rad: float, speed_mps: float) -> np.ndarray:
    """実機の生ステア/速度 → [-1, 1] 正規化アクション"""
    steer_norm = float(np.clip(steer_rad / REAL_STEER_MAX, -1.0, 1.0))
    speed_norm = float(np.clip(
        (speed_mps - config.MIN_SPEED) / (config.MAX_SPEED - config.MIN_SPEED) * 2.0 - 1.0,
        -1.0, 1.0
    ))
    return np.array([steer_norm, speed_norm], dtype=np.float32)


def convert_demo(input_path: str, output_dir: str) -> str:
    """
    実機 .npz → (observations, actions) .npz に変換して保存。

    Returns
    -------
    output_path : str
    """
    print(f"変換中: {input_path}")
    data = np.load(input_path)
    scans  = data["scans"]   # (N, M)
    steers = data["steers"]  # (N,)
    speeds = data["speeds"]  # (N,)
    N = len(steers)
    print(f"  ステップ数: {N}")

    # ─── obs ベクトルの単フレーム次元を確認 ──────────────────
    sample_obs = preprocess_lidar(scans[0])
    single_dim = len(sample_obs)
    total_dim  = single_dim * config.FRAME_STACK

    observations = np.zeros((N, total_dim), dtype=np.float32)
    actions      = np.zeros((N, 2),         dtype=np.float32)

    # ─── フレーム積層バッファ ─────────────────────────────────
    from collections import deque
    buf = deque(maxlen=(config.FRAME_STACK - 1) * config.FRAME_SKIP + 1)
    prev_action = np.zeros(2, dtype=np.float32)

    for i in range(N):
        raw_lidar = scans[i]     # (M,)
        steer_rad = float(steers[i])
        speed_mps = float(speeds[i])

        action = normalize_action(steer_rad, speed_mps)

        # 前処理 (1 フレーム分)
        frame = preprocess_lidar(raw_lidar)

        # vehicle_state と action_history を action で上書き
        offset = LIDAR_BEAMS // config.LIDAR_DOWNSAMPLE_FACTOR

        if config.INCLUDE_LIDAR_RESIDUAL:
            offset += LIDAR_BEAMS // config.LIDAR_DOWNSAMPLE_FACTOR

        if config.INCLUDE_VEHICLE_STATE:
            # speed_norm / steer_norm を埋める
            spd_norm = float(np.clip(speed_mps / config.MAX_SPEED, 0.0, 1.0))
            frame[offset]   = spd_norm
            frame[offset+1] = float(np.clip(steer_rad / REAL_STEER_MAX, -1.0, 1.0))
            offset += 2

        if config.INCLUDE_EXTRA_FEATURES:
            offset += 3

        if config.INCLUDE_RACING_LINE:
            offset += 4

        if config.INCLUDE_ACTION_HISTORY:
            frame[offset]   = prev_action[0]
            frame[offset+1] = prev_action[1]

        # フレーム積層
        buf.append(frame)
        stacked = []
        for k in range(config.FRAME_STACK):
            idx = -(k * config.FRAME_SKIP + 1)
            stacked.append(buf[idx] if abs(idx) <= len(buf) else buf[0])
        observations[i] = np.concatenate(stacked)
        actions[i]      = action
        prev_action     = action.copy()

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{N} 変換済み")

    # ─── 保存 ─────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"converted_{base}.npz")
    np.savez_compressed(output_path,
                        observations=observations,
                        actions=actions)

    print(f"  ✅ 保存: {output_path}  "
          f"(obs: {observations.shape}, actions: {actions.shape})")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="実機デモ → シム互換変換")
    parser.add_argument("--input",  type=str, default=None,
                        help="変換する .npz ファイル (単体指定)")
    parser.add_argument("--all",    action="store_true",
                        help="demos/ 以下の全 demo_*.npz を一括変換")
    parser.add_argument("--input-dir", type=str,
                        default=os.path.join(os.path.dirname(
                            os.path.dirname(os.path.abspath(__file__))), "demos"),
                        help="入力ディレクトリ (--all 使用時)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="出力ディレクトリ (未指定: 入力ディレクトリと同じ)")
    args = parser.parse_args()

    # 出力先
    out_dir = args.output_dir or args.input_dir

    if args.all:
        pattern = os.path.join(args.input_dir, "demo_*.npz")
        files   = sorted(glob.glob(pattern))
        if not files:
            print(f"[WARN] 変換対象ファイルが見つかりません: {pattern}")
            sys.exit(1)
        for f in files:
            convert_demo(f, out_dir)
    elif args.input:
        if not os.path.exists(args.input):
            print(f"[ERROR] ファイルが見つかりません: {args.input}")
            sys.exit(1)
        convert_demo(args.input, out_dir)
    else:
        parser.print_help()
        print("\n[ERROR] --input か --all を指定してください。")
        sys.exit(1)

    print("\n変換完了。次のステップ:")
    print("  python3 scripts/pretrain_bc.py --algo td3")


if __name__ == "__main__":
    main()
