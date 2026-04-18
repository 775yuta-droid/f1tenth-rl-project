# 🏎️ F1Tenth AI Racing Project
**Deep Reinforcement Learning × LiDAR-based Autonomous Racing**

F1Tenthシミュレータ上で、**LiDARセンサーのみ**を頼りに高速かつ安定した自律走行を実現するAI（PPO）を開発するプロジェクトです。

---

## 📋 プロジェクト概要

本プロジェクトでは、段階的な実験（EXP-01〜EXP-30）を通じて、F1Tenth車両の限界性能を引き出す学習を進めています。

### 🎯 技術的アプローチ
- **LiDARダウンサンプリング**: 1080点 → 108点/216点に圧縮。計算負荷を抑えつつエッジを保持。
- **フレーム積層 (Frame Stacking)**: 複数フレームを重ねることで、時間的な情報の変化（接近速度等）をAIに認識させる。
- **物理エンジンへの介入**: 車両寸法や最低速度制限（MIN_SPEED）を実機に合わせ、シミュレーションと現実の乖離を最小化。

---

## 📁 プロジェクト構成

```text
f1tenth-rl-project/
├── src/
│   ├── f1_env.py              # F1Tenth Gym 環境ラッパー
│   ├── rewards.py             # 報酬計算ロジック（ステア連動ペナルティ等の最新版）
│   └── config.py              # ⭐ 核となる全体設定ファイル
├── scripts/
│   ├── run_expXX.sh           # 実験ごとの一括実行スクリプト（学習・評価・動画）
│   ├── train.py               # 学習スクリプト（--resume 対応）
│   ├── evaluate.py            # 評価スクリプト（CSV/JSON保存）
│   ├── enjoy_wide.py          # 走行軌跡表示ビジュアライザ
│   └── view_spawn.py          # スポーン位置確認ツール
├── EXPERIMENT_PLAN.md         # 📓 実験マスタープラン・唯一の真実 (Source of Truth)
├── EXPERIMENT_REPORT.md       # 📈 歴史的な実験推移と得られた知見
├── models/                    # 学習済みモデル（.zip / .onnx）
└── logs/                      # ログ（exp29_output.log 等）
```

---

## 🚀 実験ワークフロー

本プロジェクトでは、再現性を担保するためにシェルスクリプトによる自動化を推奨しています。

### 1. 新しい実験の開始（例：EXP-30）
`config.py` や `rewards.py` を調整した後、専用のスクリプトを作成・実行します。
```bash
# 実験用スクリプトの作成（既存のものをコピーして改変）
cp scripts/run_exp29.sh scripts/run_exp30.sh

# 実行（学習、評価、動画生成が自動で走ります）
bash scripts/run_exp30.sh
```

### 2. 進捗の確認
```bash
# リアルタイムログ監視
tail -f logs/exp30_output.log

# TensorBoardでの確認
tensorboard --logdir logs --host localhost
```

---

## 💡 これまでの重要知見 (Key Findings)

実験を通じて蓄積された、学習を成功させるための「鉄則」です。

1.  **物理制限の壁 (知見12/13)**: `MIN_SPEED=1.0` ではコーナーを曲がりきれない。物理的に不可能なタスクをAIに強いると「直進して衝突」などの局習解に陥るため、適切な `MIN_SPEED`（0.3〜0.5）の設定が必須。
2.  **行動の発散 (知見2)**: `log_std_init` を適切に設定しないと、行動のばらつき（std）が20を超えて発散（パニック状態）し、完走率が0%になる。初期値は `-1.0` 以下を推奨。
3.  **Resume（継続学習）の威力**: 完走100%を達成した EXP-25 などの成功モデルから、速度設定のみを上げて Resume することで、曲がる能力を維持したまま高速化が可能。
4.  **観測次元の整合性**: `DOWNSAMPLE_FACTOR` と `FRAME_STACK` の積が変わると観測次元が変わり、モデルのロードができなくなる。Resume時は必ずこれらの値を一致させること。

---

## 🛠️ 設定ファイルの優先順位

1.  **EXPERIMENT_PLAN.md**: 過去の全実験の結果と、次に何をすべきかが記された「脳」です。書き換える前に必ず読み、実行後は必ず結果を追記してください。
2.  **src/config.py**: ステップ数、速度、スタック数など全ての物理パラメータを管理します。
3.  **src/rewards.py**: スピード、安全、進捗の重み付けを定義します。

---

## 📝 バージョン情報
- **最終更新**: 2026-04-18
- **最新実験フェーズ**: フェーズ5（Resumeによる安定高速化への挑戦）
- **対応環境**: Ubuntu 20.04/22.04 + Docker + SB3 (PPO)
