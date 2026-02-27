# 🏎️ F1Tenth AI Racing Project
**Deep Reinforcement Learning × LiDAR-based Autonomous Racing**

F1Tenthシミュレータ上で、**LiDARセンサーのみ**を頼りに自律走行するAI（PPO）を開発するプロジェクトです。

---

## 📋 プロジェクト概要

### 🎯 主要な特徴

- **LiDARダウンサンプリング**: 1080点 → 540点に圧縮。ノイズを抑制し学習を高速化
- **車両状態統合**: 速度・ステアリング角を観測に含め、マシンの慣性を考慮
- **残差処理**: 前ステップとの差分を観測に含め、動的な環境変化に対応

---

## 🛠️ 技術スタック

| 項目 | 採用技術 |
|------|---------|
| 環境 | [F1Tenth Gym](https://github.com/f1tenth/f1tenth_gym) |
| アルゴリズム | PPO (Stable Baselines3) |
| 実行環境 | Docker |
| 言語 | Python 3.9+ |

---

## 📁 プロジェクト構成

```
f1tenth-rl-project/
├── src/
│   └── f1_env.py              # F1Tenth Gym 環境ラッパー
├── scripts/
│   ├── config.py              # ⭐ 全体設定ファイル
│   ├── train.py               # 学習スクリプト
│   ├── evaluate.py            # 評価スクリプト
│   ├── enjoy_wide.py          # マップ上に走行軌跡を表示するビジュアライザ
│   ├── verify_workflow.py     # 環境動作確認スクリプト
│   ├── view_spawn.py          # スポーン位置確認ツール
│   ├── view_all_spawns.py     # 全スポーン位置を一括表示
│   ├── utils/
│   │   ├── read_logs.py       # TensorBoard ログ解析
│   │   └── read_tfevents.py   # TFEvents ファイル解析（依存なし）
│   └── tests/
│       ├── test_cleanup.py    # 環境・import の動作確認
│       ├── test_calibration.py # 観測正規化パラメータの収集
│       └── test_normalization.py # 正規化の動作確認
├── my_maps/                   # カスタムマップ
├── models/                    # 学習済みモデル（.gitignore対象）
├── logs/                      # TensorBoard ログ（.gitignore対象）
├── gif/                       # 出力動画（.gitignore対象）
├── sharing/                   # チーム内共有資料
├── Dockerfile                 # Dockerイメージ定義
└── requirements.txt           # Python依存ライブラリ
```

---

## ⚙️ 前提環境

### ハードウェア要件

| 項目 | 最小 | 推奨 |
|------|------|------|
| CPU | 4コア | 8コア+ (i7/Ryzen7) |
| RAM | 8GB | 16GB+ |
| GPU | なし | NVIDIA RTX 3060+ (6GB) |
| ストレージ | 50GB | 100GB |

### ソフトウェア

```bash
docker --version          # 20.10+
docker compose version    # 2.0+
git --version             # 2.30+
nvidia-smi                # GPU確認（あれば）
```

---

## 🚀 クイックスタート

### 1. クローン & ディレクトリ移動

```bash
git clone https://github.com/775yuta-droid/f1tenth-rl-project.git
cd f1tenth-rl-project
```

### 2. Dockerイメージのビルド

```bash
docker compose build
```

### 3. コンテナ起動

```bash
docker compose up -d
```

### 4. コンテナに接続

```bash
docker compose exec f1-sim-latest bash
# 以降のコマンドはコンテナ内で実行
```

> [!NOTE]
> CPUのみの環境では `scripts/config.py` の `DEVICE = "cuda"` を `"cpu"` に変更してください。

### 5. 動作確認

```bash
# コンテナ内 /workspace で実行
python3 scripts/verify_workflow.py
```

---

## 📚 主要スクリプト

### 学習

```bash
# コンテナ内で実行
python3 scripts/train.py --steps 500000

# 継続学習（--resume でモデルを指定）
python3 scripts/train.py --steps 500000 --resume models/checkpoints/my_model_1000000_steps
```

### 評価

```bash
python3 scripts/evaluate.py --episodes 10 --model models/my_model
```

### ビジュアライザ（走行映像の生成）

```bash
# マップ上に走行軌跡を描画（MP4 or GIF を出力）
python3 scripts/enjoy_wide.py --steps 1500 --save gif/output.mp4
```

### TensorBoard でログ確認

```bash
# ホスト側で実行
tensorboard --logdir logs --bind_all
# ブラウザで http://localhost:6006 を開く
```

### スポーン位置の確認

```bash
python3 scripts/view_spawn.py          # 現在のスポーン位置を画像に出力
python3 scripts/view_all_spawns.py     # 全スポーン位置を一括表示
```

---

## ⚙️ 設定ファイル (scripts/config.py)

### マップの切り替え

```python
# 使用するマップを変更する（拡張子なしのパス）
MAP_PATH = '/opt/f1tenth_gym/gym/f110_gym/envs/maps/berlin'

# 利用可能なマップ例
# - berlin       : 市街地コース風
# - levine       : 廊下型（基本マップ）
# - skirk        : テストコース型
# - vegas        : ラスベガス風
# - stata_basement: 複雑な地下通路
# - /workspace/my_maps/my_map : カスタムマップ
```

> [!IMPORTANT]
> マップを変更した場合は、観測正規化パラメータの再取得を推奨します:
> ```bash
> python3 scripts/tests/test_calibration.py
> ```
> 出力された値を `config.py` の `LIDAR_MEAN` 等に反映してください。

### 学習パラメータ

| パラメータ | 推奨値 | 説明 |
|-----------|--------|------|
| `TOTAL_TIMESTEPS` | 300,000〜1,500,000 | 総学習ステップ数 |
| `LEARNING_RATE` | `1e-4` | 学習率 |
| `NET_ARCH` | `[128, 128]` | ネットワーク構造 |

```python
# テスト用（数分）
TOTAL_TIMESTEPS = 50_000
NET_ARCH = [64, 64]

# 標準学習（数時間）
TOTAL_TIMESTEPS = 5_000_000
NET_ARCH = [128, 128]

# 高精度（10時間+）
TOTAL_TIMESTEPS = 25_000_000
NET_ARCH = [256, 256]
```

### 観測空間パラメータ

| パラメータ | 推奨値 | 説明 |
|-----------|--------|------|
| `LIDAR_DOWNSAMPLE_FACTOR` | `2` | 1080 → 540点に削減 |
| `INCLUDE_VEHICLE_STATE` | `True` | 速度・ステアリング角を観測に含める |
| `INCLUDE_LIDAR_RESIDUAL` | `True` | LiDAR差分を観測に含める |
| `NORMALIZE_OBSERVATIONS` | `True` | 観測値を正規化する |

### 報酬設計

```python
REWARD_COLLISION = -2000.0   # 衝突ペナルティ
REWARD_SURVIVAL  = 0.02      # ステップごとの生存報酬
REWARD_FRONT_WEIGHT = 3.0    # 前方空きスペースへの報酬
REWARD_SPEED_WEIGHT = 1.0    # 速度ボーナス
```

---

## 🐛 トラブルシューティング

### CUDA out of memory

```bash
nvidia-smi  # GPU使用状況確認
# config.py でネットワークを小さくする
# NET_ARCH = [64, 64]
```

### ObservationSpace mismatch

学習時と推論時で `config.py` の観測設定が異なると発生します。

```bash
# モデルが期待する観測次元を確認
python3 -c "
from stable_baselines3 import PPO
model = PPO.load('models/my_model')
print(f'Model expects: {model.observation_space.shape}')
"
```

`LIDAR_DOWNSAMPLE_FACTOR` / `INCLUDE_LIDAR_RESIDUAL` / `INCLUDE_VEHICLE_STATE` の設定を学習時と一致させてください。

### モデルがうまく動かない

```bash
python3 scripts/evaluate.py --episodes 5 --model models/my_model
```

結果の衝突率・平均報酬を確認し、TensorBoard のログと照合してください。

---

## 📖 実装リファレンス

### 観測空間の構成

```
obs = [
    lidar_downsampled,                    # 540次元 (1080 / 2)
    lidar_downsampled - prev_lidar,       # 540次元 (残差, INCLUDE_LIDAR_RESIDUAL=True時)
    [speed_normalized, steering_angle],   # 2次元 (INCLUDE_VEHICLE_STATE=True時)
]
# 合計: 1082次元 (デフォルト設定)
```

### アクション空間

```
action = [steering, throttle]   # どちらも [-1.0, +1.0] の正規化空間

actual_steering = steering * STEER_SENSITIVITY
actual_speed    = MIN_SPEED + (throttle + 1.0) * (MAX_SPEED - MIN_SPEED) / 2.0
```

### TensorBoard メトリクス

| メトリクス | 意味 | 良い傾向 |
|-----------|------|---------|
| `train/loss` | ポリシー損失 | 低下 ✅ |
| `train/value_loss` | 価値関数損失 | 低下 ✅ |
| `train/entropy_loss` | 探索度 | 0.1〜0.3 ✅ |
| `train/explained_variance` | 価値関数精度 | 0.8以上 ✅ |

---

## 🚀 関連ドキュメント

- [改善プラン](./sharing/plan.md) — Jetson実走行向け改善
- [モデル構造説明](./sharing/model_structure.md) — MLPアーキテクチャ
- [Jetsonデプロイ計画](./sharing/JETSON_DEPLOYMENT_PLAN.md) — 実機統合

---

## 📚 参考リソース

- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [F1Tenth Gym](https://github.com/f1tenth/f1tenth_gym)
- [PPO 論文](https://arxiv.org/abs/1707.06347)

---

## 📝 バージョン情報

- **作成**: 2026-02-24
- **最終更新**: 2026-02-27
- **バージョン**: 1.1.0
- **対応 Python**: 3.9+

---

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

Licensed under the MIT License. See [LICENSE](./LICENSE) for details.
