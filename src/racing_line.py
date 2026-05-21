"""
レーシングライン（中心線ウェイポイント）管理モジュール

generate_centerline.py で生成した CSV を読み込み、
エージェントの観測に追加する特徴量を計算します。

提供する特徴量:
    - cte        : 中心線からの横方向誤差 [m]  （正=左ずれ, 負=右ずれ）
    - heading_err: コース方向とのヘディング誤差 [rad]
    - curvature  : 前方 N 点の平均曲率 [1/m]
    - progress   : ウェイポイントインデックスの進捗割合 [0, 1]
"""

import numpy as np
import csv
import os
from typing import Optional, Tuple


class RacingLine:
    """
    CSVウェイポイントからレーシングライン特徴量を計算するクラス。

    Args:
        csv_path: generate_centerline.py が出力した CSV ファイルのパス。
                  存在しない場合は空ラインとして動作（全特徴量=0）。
        lookahead: 前方曲率を計算するウェイポイント数
    """

    NUM_FEATURES = 4  # [cte, heading_err, curvature, progress]

    @property
    def num_waypoints(self) -> int:
        return len(self.xy)


    def __init__(self, csv_path: str, lookahead: int = 12):  # [Fix-Curve4] 5→12: S字第2カーブを曲率特徴量に先読み
        self.csv_path  = csv_path
        self.lookahead = lookahead
        self._loaded   = False

        self.xy        = np.zeros((0, 2), dtype=np.float32)
        self.heading   = np.zeros(0,      dtype=np.float32)
        self.curvature = np.zeros(0,      dtype=np.float32)

        # 最後に見つけた最近傍インデックス（高速サーチ用）
        self._last_idx = 0

        self._load(csv_path)

    # ------------------------------------------------------------------
    def _load(self, csv_path: str) -> None:
        if not os.path.exists(csv_path):
            print(f"[RacingLine] CSV が見つかりません: {csv_path}")
            print(f"[RacingLine] scripts/utils/generate_centerline.py を実行して生成してください。")
            print(f"[RacingLine] フォールバック: 全特徴量=0 で動作します。")
            return

        xs, ys, hs, ks = [], [], [], []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                xs.append(float(row["x"]))
                ys.append(float(row["y"]))
                hs.append(float(row["heading"]))
                ks.append(float(row["curvature"]))

        self.xy        = np.column_stack([xs, ys]).astype(np.float32)
        self.heading   = np.array(hs, dtype=np.float32)
        self.curvature = np.array(ks, dtype=np.float32)

        # 骨格化ノイズ由来の外れ値をクリップ（±10 [1/m] 以上は除去）
        # F1Tenth スケール（数m幅）のコースで曲率 10 以上は物理的に非現実的
        CURV_CLIP = 10.0
        self.curvature = np.clip(self.curvature, -CURV_CLIP, CURV_CLIP)

        self._loaded   = True
        self._last_idx = 0
        print(f"[RacingLine] ロード完了: {csv_path} ({len(self.xy)} ウェイポイント)")
        print(f"[RacingLine] 曲率: min={self.curvature.min():.3f}, max={self.curvature.max():.3f}, |mean|={np.abs(self.curvature).mean():.3f} [1/m]")

    # ------------------------------------------------------------------
    def _find_nearest(self, x: float, y: float) -> int:
        """
        現在位置 (x, y) に最も近いウェイポイントのインデックスを返す。
        前回インデックスの周辺（±50点）を優先探索して高速化。
        """
        if not self._loaded:
            return 0

        N = len(self.xy)
        half_window = 50
        lo = max(0, self._last_idx - half_window)
        hi = min(N, self._last_idx + half_window + 1)

        window = self.xy[lo:hi]
        # np.linalg.norm (平方根) ではなく二乗距離で比較して高速化
        diff   = window - np.array([x, y], dtype=np.float32)
        dists_sq = np.sum(diff**2, axis=1)
        local_idx = int(np.argmin(dists_sq))
        idx = lo + local_idx

        # ウィンドウ端に近い場合は全探索にフォールバック
        if local_idx < 5 or local_idx > (hi - lo - 5):
            diff_all = self.xy - np.array([x, y], dtype=np.float32)
            dists_all_sq = np.sum(diff_all**2, axis=1)
            idx = int(np.argmin(dists_all_sq))

        self._last_idx = idx
        return idx

    def get_nearest_index(self, x: float, y: float) -> int:
        """現在位置に最も近いウェイポイントのインデックスを返す"""
        return self._find_nearest(x, y)

    # ------------------------------------------------------------------
    def get_features(
        self,
        x: float,
        y: float,
        yaw: float,
        max_cte: float = 2.0,
        max_curv: float = 10.0,   # _load時のCURV_CLIPに合わせる
    ) -> np.ndarray:
        """
        エージェントの現在状態から観測特徴量を計算する。

        Args:
            x, y : ワールド座標 [m]
            yaw  : ヘディング角 [rad]
            max_cte : CTE の正規化上限 [m]
            max_curv: 曲率の正規化上限 [1/m]

        Returns:
            features: (4,) float32 配列、値域はおよそ [-1, 1]
                [0] cte_norm        : 中心線からの横方向誤差（正規化）
                [1] heading_err_norm: ヘディング誤差（正規化、/π）
                [2] curvature_norm  : 前方平均曲率（正規化）
                [3] progress        : コース進捗割合 [0, 1]
        """
        if not self._loaded:
            return np.zeros(self.NUM_FEATURES, dtype=np.float32)

        idx = self._find_nearest(x, y)
        N   = len(self.xy)

        # ---- CTE (Cross-Track Error) ----
        wp       = self.xy[idx]
        wp_head  = self.heading[idx]

        # 車両→ウェイポイントベクトルをコース方向に投影
        dx = x - wp[0]
        dy = y - wp[1]
        # コース法線方向 (ヘディングに垂直、左が正)
        cte = -dx * np.sin(wp_head) + dy * np.cos(wp_head)
        cte_norm = float(np.clip(cte / max_cte, -1.0, 1.0))

        # ---- Heading Error ----
        head_err = yaw - wp_head
        # [-π, π] に収める
        head_err = (head_err + np.pi) % (2 * np.pi) - np.pi
        head_err_norm = float(np.clip(head_err / np.pi, -1.0, 1.0))

        # ---- 前方曲率 ----
        lo_k = idx
        hi_k = min(N, idx + self.lookahead)
        mean_curv = float(np.mean(self.curvature[lo_k:hi_k]))
        curv_norm = float(np.clip(mean_curv / max_curv, -1.0, 1.0))

        # ---- 進捗割合 ----
        progress = idx / float(N - 1) if N > 1 else 0.0

        return np.array([cte_norm, head_err_norm, curv_norm, progress], dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """エピソードリセット時に呼び出す（最近傍サーチ位置をリセット）"""
        self._last_idx = 0
