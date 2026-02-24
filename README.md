# 🏎️ F1Tenth AI Racing Project
**Deep Reinforcement Learning × LiDAR-based Autonomous Racing**

F1Tenthシミュレータ上で、**LiDARセンサーのみ**を頼りに自律走行するAI（PPO）を開発するプロジェクトです。

---

## 📋 プロジェクト概要

### 🎯 主要な特徴

**「壁の隙間を道と誤認する」** という強化学習特有の課題を以下で解決：

- **LiDARダウンサンプリング**: 1080点 → 540点に圧縮。ノイズを抑制し学習を高速化
- **車両状態統合**: 速度・ステアリング角を観測に含め、マシンの慣性を考慮
- **残差処理**: 前ステップとの差分を観測に含め、動的な環境変化に対応

---

## 🛠️ 技術スタック

| 項目 | 採用技術 |
|------|---------|
| 環境 | [F1Tenth Gym](https://github.com/f1tenth/f1tenth_gym) |
| アルゴリズム | PPO (Stable Baselines3) |
| 実行環境 | Docker (GPU対応) |
| 言語 | Python 3.9+ |

---

## 📁 プロジェクト構成

```
f1tenth-rl-project/
├── src/f1_env.py              # F1Tenth環境ラッパー
├── scripts/
│   ├── train.py               # 学習スクリプト
│   ├── evaluate.py            # 評価スクリプト
│   ├── enjoy.py               # ビジュアライザ（LiDAR版）
│   ├── enjoy-wide.py          # ビジュアライザ（マップ版）
│   ├── config.py              # 重要: 全体設定ファイル
│   ├── verify_workflow.py     # 環境検証
│   └── read_logs.py           # ログ解析
├── models/                    # 学習済みモデル
├── my_maps/                   # カスタムマップ
└── sharing/                   # チーム共有ドキュメント
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

**GPU環境の場合**:
```bash
docker compose up -d
docker compose run f1-sim-latest bash
```

**CPU環境の場合** (GPU なし):
```bash
sed -i 's/DEVICE = "cuda"/DEVICE = "cpu"/' scripts/config.py
docker compose up -d
docker compose run f1-sim-latest bash
#CPUが一番速い
```

### 4. 動作確認

```bash
cd scripts
python3 verify_workflow.py
```

---

### 学習設定別

```python
# テスト用（30分、GPU必須）
TOTAL_TIMESTEPS = 50,000
NET_ARCH = [64, 64]

# 標準学習（3-4時間推奨）
TOTAL_TIMESTEPS = 500,000
NET_ARCH = [128, 128]

# 高精度（8時間+）
TOTAL_TIMESTEPS = 1,500,000
NET_ARCH = [256, 256]
```

---

## 📚 主要スクリプト

### 学習

```bash
cd scripts
python3 train.py --steps 500000 --model ../models/my_model
```

### 評価

```bash
python3 evaluate.py --episodes 10 --model ../models/my_model
```

### ビジュアライザ

```bash
python3 enjoy.py          # LiDAR点群表示
python3 enjoy-wide.py     # マップ上に走行軌跡表示
```

### ログ確認

```bash
tensorboard --logdir=../logs --port=6006
```

---

## ⚙️ 設定ファイル (scripts/config.py)

### 観測空間パラメータ

**LIDAR_DOWNSAMPLE_FACTOR** (推奨: 2)
- 1080 → 540次元に圧縮
- ✅ 計算 50% 高速化 | ❌ 細かい障害物検出低下
- `1`: 高精度、`2`: バランス、`4`: 高速

**INCLUDE_LIDAR_RESIDUAL** (推奨: True)
- 前ステップとの差分を含める
- ✅ 衝突回避精度向上

**LIDAR_MEAN / LIDAR_STD** (重要)
- LiDAR の正規化パラメータ
- マップ変更時に再取得:
  ```bash
  python3 calibrate_stats.py
  ```

### 学習パラメータ

**LEARNING_RATE** (推奨: 1e-4)
- `1e-3`: 高速だが不安定
- `1e-4`: バランス型 ✅
- `1e-5`: 安定だが遅い

**NET_ARCH** (推奨: [128, 128])
- `[64, 64]`: 軽い・高速
- `[128, 128]`: バランス ✅
- `[256, 256]`: 高精度・重い

### 報酬設計

```python
REWARD_COLLISION = -2000.0      # 衝突ペナルティ
REWARD_SURVIVAL = 0.02           # ステップごと基本報酬
REWARD_FRONT_WEIGHT = 3.0        # 前方空きスペース
REWARD_SPEED_WEIGHT = 1.0        # 速度ボーナス
```

---

## 🐛 トラブルシューティング

### CUDA out of memory

```bash
nvidia-smi

# メモリ削減
sed -i "s/NET_ARCH = \[128, 128\]/NET_ARCH = [64, 64]/" scripts/config.py
```

### ObservationSpace mismatch

原因: 学習時と推論時で config.py が異なる

```bash
python3 << 'EOF'
from stable_baselines3 import PPO
model = PPO.load("../models/my_model")
print(f"Model expects: {model.observation_space.shape}")
EOF
```

### Model performs poorly

```bash
python3 << 'EOF'
from stable_baselines3 import PPO
import config
from src.f1_env import F1TenthRL

env = F1TenthRL(config.MAP_PATH)
model = PPO.load("../models/my_model")

obs = env.reset()
total_reward = 0
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        break

print(f"Total Reward: {total_reward}")
EOF
```

---

## 📖 実装リファレンス

### 観測空間の構成

```python
obs = np.concatenate([
    lidar_downsampled,              # 540次元
    lidar_downsampled - prev_lidar, # 540次元（残差）
    [speed, steering_angle],        # 2次元
])
# 合計: 1082次元
```

### アクション空間

```python
action = [steering, speed]  # [-1, +1] 正規化空間

actual_steering = steering * 0.4 rad
actual_speed = speed * MAX_SPEED (1-3 m/s)
```

### TensorBoard メトリクス

| メトリクス | 意味 | 目安 |
|-----------|------|------|
| train/loss | ポリシー損失 | 低下傾向 ✅ |
| train/value_loss | 価値関数損失 | 低下傾向 ✅ |
| train/entropy_loss | 探索度 | 0.1～0.3 ✅ |
| train/explained_variance | 価値関数精度 | 0.8+ ✅ |

---

## 🚀 次のステップ

### チーム内共有ドキュメント

- [改善プラン](./sharing/plan.md) - Jetson実走行向け改善
- [モデル構造説明](./sharing/model_structure.md) - MLPアーキテクチャ
- [Jetsonデプロイ計画](./sharing/JETSON_DEPLOYMENT_PLAN.md) - 実機統合

---

## 👥 コントリビューション

### Issue報告

```markdown
### 環境情報
- OS: WSL2 Ubuntu 20.04
- GPU: RTX 5060
- コマンド: python3 train.py --steps 100000

### エラー内容
[ここにエラーメッセージを貼り付け]
```

### PRガイドライン

```bash
git checkout -b feature/add-lstm-encoder
git commit -m "[feature] Add LSTM encoder

- 20% reward improvement
- Tested on Levine map"
```

---

## 📝 ファイル情報

- **作成**: 2026-02-24
- **バージョン**: 1.0.0
- **対応 Python**: 3.9+

---

## 📚 参考リソース

- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [F1Tenth Gym](https://github.com/f1tenth/f1tenth_gym)
- [PPO 論文](https://arxiv.org/abs/1707.06347)
```
---
