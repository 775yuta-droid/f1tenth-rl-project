# 🏎️ F1Tenth AI Racing Project
**Deep Reinforcement Learning × LiDAR-based Autonomous Racing**

F1Tenthシミュレータ上で、**LiDARセンサーのみ**を頼りに高速かつ安定した自律走行を実現するAI（PPO）を開発するプロジェクトです。

---

## 🚀 クイックスタート (Docker)

別のPCへ移行した後、以下の手順で最短で実験を再開できます。

### 1. 環境の起動
```bash
# イメージのビルドとコンテナの起動
docker compose up -d

# コンテナ内に入る
docker compose exec f1-sim-latest bash
```

### 2. 学習の実行 (EXP-34の例)
```bash
# 実験用ディレクトリへ移動し実行
bash scripts/experiments/run_exp34.sh
```

### 3. 進捗確認
別のターミナルで TensorBoard を起動してブラウザ（`localhost:6006`）で確認できます。
```bash
tensorboard --logdir logs --host localhost
```

---

## 📋 プロジェクト概要

本プロジェクトでは、段階的な実験を通じて、F1Tenth車両の限界性能を引き出す学習を進めています。

### 🎯 技術的アプローチ
- **LiDARダウンサンプリング**: 1080点 → 108点に圧縮。さらに実機（Hokuyo）に合わせ観測範囲を **270°** に制限することで Sim-to-Real ギャップを最小化。
- **フレーム積層 (Frame Stacking)**: 複数フレームを重ねることで、時間的な情報の変化（接近速度等）をAIに認識させる。
- **物理エンジンへの介入**: 車両寸法や最低速度制限（`MIN_SPEED`）を実機に合わせ、シミュレーションと現実の乖離を最小化。

---

## 📁 プロジェクト構成

```text
f1tenth-rl-project/
├── src/                # ⭐ コアロジック
│   ├── f1_env.py       # F1Tenth Gym 環境ラッパー
│   ├── rewards.py      # 報酬計算ロジック
│   ├── config.py       # 全体設定
│   └── profiles.py     # PC別ハードウェア設定
├── scripts/            # 🛠️ ツール
│   ├── experiments/    # 🧪 個別実験スクリプト (run_expXX.sh)
│   ├── train.py        # 学習メイン
│   ├── evaluate.py     # 評価（CSV/JSON保存）
│   └── enjoy_wide.py   # 動画/軌跡生成
├── my_maps/            # 🗺️ カスタムマップ (.pgm/.yaml)
├── EXPERIMENT_PLAN.md   # 📓 実験マスタープラン (Source of Truth)
├── EXPERIMENT_REPORT.md # 📈 実験履歴と知見集
├── models/             # 💾 学習済みモデル (.zip) ※Git管理外
└── logs/               # 📊 TensorBoardログ ※Git管理外
```

---

## ⚙️ 主要な環境変数
`config.py` を通じて、以下の環境変数で動作を制御できます。

| 変数名 | 説明 | 設定例 |
| :--- | :--- | :--- |
| `TRAINING_PROFILE` | ハードウェア設定 | `laptop` / `desktop` / `auto` |
| `MAP_PATH` | マップのパス（拡張子なし） | `/workspace/my_maps/testmap-0416` |
| `MIN_SPEED` | 車両の最低速度制限 | `0.3` |
| `MAX_SPEED` | 車両の最高速度制限 | `2.0` |

---

## 📦 モデルの管理と移行
モデルファイル（`models/*.zip`）はファイルサイズが大きいため Git 管理から除外されています。別のPCへ移行する場合は、以下のファイルを優先的に手動コピーしてください。

1.  **`ppo_10M_exp25_fast_stable.zip`**: 広域マップ完走100%のベース。
2.  **`ppo_exp33b_p3_narrow_normal.zip`**: 狭域マップ適応済みのベース。

---

## 🛠️ ONNX変換 (実機デプロイ用)

学習したモデルを実機（Jetson等）で軽量に動作させるため、ONNX形式への変換をサポートしています。

### 1. ONNX形式への変換
`scripts/export/convert_ppo_to_onnx.py` を使用して、SB3の `.zip` モデルを `.onnx` に変換します。
```bash
python scripts/export/convert_ppo_to_onnx.py --model models/ppo_expXX.zip
```
※ デフォルトでは同じディレクトリに `.onnx` ファイルが生成されます。`--output` で出力先を指定可能です。

### 2. 変換の検証
変換後のモデルが元のモデルと同じ推論結果を出すか確認します。
```bash
python scripts/export/verify_onnx.py --model models/ppo_expXX.zip --output models/ppo_expXX.onnx
```
`Success: ONNX outputs match SB3 outputs!` と表示されれば成功です。

### ⚠️ 注意点
- **観測次元**: 変換時の観測次元（LiDAR点数 + 状態数）は学習時の設定に依存します。実機の推論コード側でも、入力次元を一致させる必要があります（例：108点LiDAR + 2点State = 110次元）。
- **決定論的推論**: 実機デプロイ用のONNXモデルは、常に決定論的な行動（Deterministic Action）を出力するようにエクスポートされます。

---

## 💡 これまでの重要知見 (Key Findings)

実験を通じて蓄積された、学習を成功させるための「鉄則」です。

1.  **物理制限の壁**: `MIN_SPEED=1.0` ではコーナーを曲がりきれない。物理的に不可能なタスクをAIに強いると「直進して衝突」などの局所解に陥るため、適切な `MIN_SPEED`（0.3〜0.5）の設定が必須。
2.  **行動の発散**: `log_std_init` を適切に設定しないと、行動のばらつき（std）が20を超えて発散（パニック状態）し、完走率が0%になる。初期値は `-1.0` 以下を推奨。
3.  **Resume（継続学習）の威力**: 完走100%を達成した EXP-25 などの成功モデルから、速度設定のみを上げて Resume することで、曲がる能力を維持したまま高速化が可能。
4.  **観測次元の整合性**: `DOWNSAMPLE_FACTOR` と `FRAME_STACK` の積、および **LiDAR のスキャン角度範囲**が変更されると観測次元が変わり、モデルのロードができなくなる。270°化への移行時は Fresh学習を推奨。

---

## 📝 バージョン情報
- **最終更新**: 2026-04-22
- **最新実験フェーズ**: フェーズ10（実機仕様への完全移行と 270° 統一）
- **対応環境**: Ubuntu 20.04 / 22.04 + Docker + SB3 (PPO)
