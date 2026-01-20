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
**報酬設計（Reward Engineering）によって解決**した点にあります。

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
### 1. vscodeを開きwslを起動する

### 2. コンテナを起動して中に入る

    docker compose up -d
    docker compose exec f1-sim bash
---
## 🛠️ 終了方法
### 1. コンテナの停止

    docker compose down

### 2. Gitに保存

    git add .
    git commit -m "feat: refine reward function and update dockerfile"
    git push origin main
---
## 📂 ファイル構成と役割

- **Dockerfile**  
  NVIDIA GPU対応、FFmpeg（動画生成）、Git safe.directory 設定済み

- **scripts/train.py**  
  学習実行スクリプト  
  路地回避のための **最小距離報酬（min_front_dist）** を実装

- **scripts/enjoy^.py**  
  評価・可視化スクリプト  
  初期版

- **scripts/enjoy-wide.py**  
  評価・可視化スクリプト  
  前方25m・左右15mの **広角視点** で走行をGIF化

- **models/**  
  学習済みPPOモデルの保存先

- **run_simulation_final.gif**  
  最新の学習結果によるデモ走行

---

## 🚀 実行方法

### 学習（Training）

    rm -f models/ppo_f1_final.zip
    python3 scripts/train.py

学習完了時に **システムベルが鳴る**。

---

### 評価と録画（Evaluation）

    python3 scripts/enjoy.py

---

## 📈 報酬設計（路地回避ロジック）

### 隙間への迷い込み対策
- 前方中央20度の **平均距離** を廃止
- **最小距離（min_front_dist）** を報酬に採用
- 細い隙間より **道幅のある本線** を選択

### 旋回性能向上
- ステアリング感度：**0.8**
- 速度固定：**2.5**

### コースアウト抑制
- 衝突ペナルティ：**-200.0**

---

## 🔧 トラブルシューティング

Docker環境で発生する  
**「dubious ownership」エラー**は、  
Dockerfile 内の `safe.directory` 設定により自動解決される。


