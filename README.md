# F1Tenth AI Racing Project
**Deep Reinforcement Learning × LiDAR**

F1Tenthシミュレータ上で、**LiDARセンサーのみ**を頼りに自律走行を行う  
AI（PPO: Proximal Policy Optimization）を開発するプロジェクトです。

---

## 🏎️ プロジェクト概要

本プロジェクトでは、AIにコース形状を学習させ、  
**壁に衝突することなく高速で周回する自律走行**を目指します。

最大の特徴は、

> **「壁の隙間を道と誤認してコース外へ迷い込む」**

という強化学習特有の課題を、  
**報酬設計（Reward Engineering）** と **観測空間の工夫（Observation Engineering）** によって解決しました。

#### 観測空間の工夫
- **LiDARダウンサンプリング**: 1080点のデータを1/10に圧縮。ノイズを抑制し、学習を高速化。
- **車両状態の統合**: 速度とステアリング角を観測に含めることで、マシンの慣性を考慮した判断が可能に。

---

## 🛠️ 環境構築（Docker / NVIDIA GPU対応）
**前提環境**
WSL環境(Ubuntu20.04L TS)

GPUを活用し、  
**学習・描画・GIF生成を安定させるための決定版Dockerfile**を使用します。

### 1. gitを利用してcloneする

    git clone https://github.com/775yuta-droid/f1tenth-rl-project.git
    cd f1tenth-rl-project

### 2. コンテナイメージをbuild

    docker compose build 

### 3. コンテナのをDockerFileを使用してbuild（GPU有効）

    docker compose up -d
    docker compose exec f1-sim bash
---

## 🛠️ 二回目からの起動方法
### 1. wslを起動する

### 2. コンテナを起動して中に入る

    docker compose up -d
    #rtx5060で学習するとき
    docker compose run f1-sim-latest bash
    #20.04を使うとき
    docker compose run f1-sim-legacy bash


---
## 🛠️ 終了方法
### 1. コンテナの停止

    docker compose down

### 2. Gitに保存

    git add .
    git commit -m "example commit"
    git push origin main
---
## 📂 ファイル構成と役割

- **Dockerfile / Dockerfile.2204**  
  NVIDIA GPU対応、FFmpeg（動画生成）、Git safe.directory 設定済み。Gymのビルドエラー回避のためのバージョン固定済み。

- **docker-compose.yml**  
  `f1-sim-latest`: 最新環境（Ubuntu 22.04ベース、RTX 5060想定）
  `f1-sim-legacy`: 旧環境（Ubuntu 20.04ベース）

- **requirements.txt**  
  Pythonパッケージの依存関係。Gym 0.23.1, Stable-Baselines3, PyYAML などのバージョンが固定されています。

- **src/f1_env.py**  
  F1Tenth環境クラス。Stable-Baselines3でそのまま使える Gym Wrapper です。

- **scripts/config.py**  
  報酬設計、物理パラメータ（最高速度など）、パス設定を一括管理する中心ファイル。

- **scripts/train.py**  
  学習実行スクリプト。

- **scripts/enjoy-wide.py**  
  **【高機能版】** 評価・可視化スクリプト。
  - レンダリング速度の最適化済み
  - **走行軌跡（ブルーのライン）** の表示
  - **LiDAR点群（シアンの点）** の可視化
  - リアルタイムHUD（速度、ステアリング、ステップごとの報酬）

- **scripts/evaluate.py**  
  **【定量評価版】** 性能計測スクリプト。
  - 複数エピソード実行して平均速度や完走率を算出
  - 報酬設計の良し悪しを数値で比較可能
  - 観測空間の次元不一致に対するエラーハンドリング済み

- **scripts/verify_workflow.py**  
  **【環境検証用】** 改修後の環境が正しく動作するかを自動チェックするスクリプト。
  - 観測空間の形状チェック、短時間学習、GIF生成までを一気通貫でテスト。

---

## 🚀 実行方法

### 1. 学習（Training）

```bash
# 学習済みモデルをリセットして開始する場合
rm -f models/*.zip
python3 scripts/train.py
```

#### 学習の監視 (TensorBoard)
学習の進捗（報酬の推移など）をブラウザでリアルタイムに確認できます。
```bash
# コンテナ内で実行
tensorboard --logdir logs --bind_all
```
ホストマシンのブラウザから `http://localhost:6006` で閲覧可能です。

### 2. 評価と録画（Visual Evaluation）

```bash
# 標準設定で実行（GIFが ../gif/ に保存されます）
python3 scripts/enjoy-wide.py

# オプション指定（ステップ数、モデル、保存先）
python3 scripts/enjoy-wide.py --steps 3000 --model models/my_best_model --save my_drive.mp4
```

| 引数 | 説明 | デフォルト |
| :--- | :--- | :--- |
| `--steps` | 最大ステップ数 | 1500 |
| `--model` | モデルのパス（拡張子なし） | config内のパス |
| `--save` | 保存先パス | config内のパス |
| `--no-render` | GIFを生成せずシミュレーションのみ | Off |

### 3. ベンチマーク実行（Quantitative Evalu ation）

```bash
# 10エピソード走らせて平均性能を計測
python3 scripts/evaluate.py --episodes 10
```

### 4. 環境の整合性チェック（Verification）

```bash
# 改修後に環境が正しく動作するか自動テスト
python3 scripts/verify_workflow.py
```

---

## 🔧 環境のリセット・再構築
ライブラリのバージョン変更を反映させるには、一度コンテナを破棄してビルドし直すのが確実です。

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```
---
## 🔧 トラブルシューティング

Docker環境で発生する  
**「dubious ownership」エラー**は、  
Dockerfile 内の `safe.directory` 設定により自動解決される。
