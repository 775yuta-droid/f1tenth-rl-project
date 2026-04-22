# F1TENTH 強化学習システム 技術整理まとめ

## ■ 概要
本プロジェクトは、LiDARセンサを用いたF1TENTH車両の自律走行を強化学習で実現する。

### 現在の構成
- センサ：Hokuyo LiDAR
- 入力：LiDAR（ダウンサンプリング）＋フレームスタック
- モデル：MLP（全結合ニューラルネット）
- アルゴリズム：PPO
- 実行環境：Jetson Orin Nano

---

## ■ PPOとは
PPO（Proximal Policy Optimization）は強化学習アルゴリズムであり、  
ニューラルネットの「学習方法」を指す。

### 構造
入力 → ニューラルネット → 行動  
　　　　　　　↑  
　　　　　 PPOで学習

---

## ■ 現在のボトルネック
- 前処理の不安定性
- 入力設計の不十分さ
- LiDAR構造未活用（MLPの限界）

※ アルゴリズム自体は問題ではない

---

## ■ LiDAR前処理（最重要）

### 問題点
- inf / NaN 未処理
- クリッピングなし
- Z-score正規化（不安定）

---

### 推奨前処理

    def preprocess_lidar(lidar, max_range=30.0):
        import numpy as np

        lidar = np.array(lidar)

        lidar = np.nan_to_num(lidar, nan=max_range)
        lidar[np.isinf(lidar)] = max_range
        lidar = np.clip(lidar, 0.0, max_range)

        lidar = lidar / max_range
        lidar = 1.0 - lidar

        return lidar

---

### 数値条件
- 範囲：0.0 ～ 1.0
- NaN：なし
- inf：なし

---

## ■ 正規化 vs 正則化

| 用語 | 内容 |
|------|------|
| 正規化 | 入力スケール調整 |
| 正則化 | 過学習防止 |

優先度：
- 正規化：必須
- 正則化：後回し

---

## ■ 入力設計

### 改善案
- 前方距離（front）
- 最小距離（min）

---

### 構成
LiDAR → 正規化 → フレームスタック → 特徴追加 → NN

---

## ■ モデル構造

### 現状
MLP（全結合）

### 問題
- 空間構造を無視

---

## ■ 改善案：Conv1D

### 構成
LiDAR → Conv1D → Conv1D → MLP → 出力

### 効果
- 壁・コーナー検出
- ノイズ耐性向上

---

## ■ 報酬設計

### 現状要素
- 前方距離
- 速度
- 壁距離
- 進行距離

---

### 注意
- 速度偏重 → 衝突
- 安全偏重 → 遅い

---

## ■ PPO以外

### SAC
- 高性能
- 不安定

### 結論
現段階ではPPOが最適

---

## ■ 優先順位

1. 前処理修正
2. 報酬調整
3. 入力設計改善
4. CNN導入
5. PPO調整
6. SAC検討

---

## ■ 結論

改善の本質は：

「アルゴリズムではなく設計」

特に重要：
- 前処理
- 入力設計

---

## ■ 次のアクション

- 前処理修正（最優先）
- CNN導入
- 再学習・比較