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

## ⚙️ 前提環境

### ハードウェア要件

| 項目 | 最小 | 推奨 |
|------|------|------|
| CPU | 4コア | 8コア+ (i7/Ryzen7) |
| RAM | 8GB | 16GB+ |
| GPU | なし | NVIDIA RTX 3060+ (6GB) |
| ストレージ | 50GB | 100GB |

---

## 🚀 クイックスタート

### 1. クローン

```bash
git clone https://github.com/775yuta-droid/f1tenth-rl-project.git
cd f1tenth-rl-project
