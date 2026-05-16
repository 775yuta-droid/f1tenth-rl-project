# F1TENTH 強化学習システム 技術整理まとめ

## ■ 概要
本プロジェクトは、LiDARセンサを用いたF1TENTH車両の自律走行を強化学習で実現する。

### 現在の構成
- センサ：Hokuyo LiDAR（270°）
- 入力：LiDAR（ダウンサンプリング）＋フレームスタック＋車両状態＋追加特徴
- モデル：**Conv1D + MLP（CNN特徴抽出器）**
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

## ■ 現在のボトルネック（解決済み）
- ~~前処理の不安定性~~ → **0-1反転正規化に修正 ✅**
- ~~入力設計の不十分さ~~ → **front_dist / min_dist 特徴を追加 ✅**
- ~~LiDAR構造未活用（MLPの限界）~~ → **Conv1D特徴抽出器を導入 ✅**
- ~~時間情報の欠落~~ → **torch.flip および サブステップ観測更新を導入 ✅**

---

## ■ LiDAR前処理（実装済み）

### 実装済み前処理（development-plan.md準拠）

```python
def preprocess_lidar(lidar, max_range=30.0):
    import numpy as np
    lidar = np.array(lidar)
    lidar = np.nan_to_num(lidar, nan=max_range)
    lidar[np.isinf(lidar)] = max_range
    lidar = np.clip(lidar, 0.0, max_range)
    lidar = lidar / max_range
    lidar = 1.0 - lidar   # ← 近い壁=1.0, 遠い空間=0.0
    return lidar
```

実装場所: `src/f1_env.py` の `_get_obs()` 内

---

## ■ 入力設計（実装済み）

### 改善案（実装済み）
- 前方距離（`front_dist`）: `1.0 - clip(min(s[285:525])) / 30.0`
- 最小距離（`min_dist`）: `1.0 - clip(min(s)) / 30.0`

`config.py`: `INCLUDE_EXTRA_FEATURES = True` で有効

---

## ■ モデル構造（最新：EXP-48構成）

LiDAR空間特徴を抽出する **Conv1D** と、車両状態を処理する **MLP** のハイブリッド構成。

```mermaid
graph TD
    subgraph Input
        L[LiDAR 216点 x 4F] -- torch.flip --> C1
        S[車両状態 2点 x 4F] --> SFC
    end

    subgraph ConvBlock["Conv1DLidarExtractor"]
        C1[Conv1D k7/p3/s1] --> M1[MaxPool /2]
        M1 --> C2[Conv1D k9/p4/s1] --> M2[MaxPool /2]
        M2 --> C3[Conv1D k5/p2/s1] --> GAP[AdaptiveAvgPool1d 16]
        GAP --> LFC[Linear 256]
        SFC[Linear 32] --> CAT
        LFC --> CAT[Concat]
        CAT --> OFC[Linear 256]
    end

    subgraph PPOHead["net_arch=[128, 128]"]
        OFC --> P[Actor Branch]
        OFC --> V[Critic Branch]
    end
```

*   **実装場所**: `src/cnn_policy.py` (`Conv1DLidarExtractor`)
*   **特徴**:
    *   **AdaptiveAvgPool1d**: 入力解像度に依存しない堅牢な設計。
    *   **時間順序の正常化**: `torch.flip` により物理的な因果関係を正しく学習。

---

## ■ 報酬設計 (最新ロジック)

### マップ最適化報酬
- **前方空間**: マップ実測値（8m）で正規化。前方+斜め（`open_side`）のハイブリッド評価。
- **3段階安全スコア**: マップの壁距離統計値（p50/p75/Danger）に基づいた動的な線形補間。
- **ブレーキ能力**: `safe_brake_dist` の定数項をマップ幅に合わせて縮小。
- **スピンペナルティ**: 報酬ハッキング（広い場所での回転）を防止。

---

## ■ 進捗状況まとめ

| # | 項目 | 状態 |
|---|------|------|
| 1 | 前処理修正 | ✅ 完了 |
| 2 | 入力設計改善 (front_dist/min_dist) | ✅ 完了 |
| 3 | CNN導入 (Conv1D) | ✅ 完了 |
| 4 | 報酬調整 (統計的正規化/スピン防止) | ✅ 完了 |
| 5 | モデル軽量化・柔軟性向上 (GAP導入) | ✅ 完了 |
| 6 | 時間軸反転バグの修正 | ✅ 完了 |
| 7 | サブステップ観測更新の実装 | ✅ 完了 |
| 8 | 時系列処理の更なる強化 (LSTM等) | ⬜ 検討中 |
| 9 | 残差強化学習 (Residual RL) | ✅ 完了 |

---

## ■ 次のフェーズ：残差強化学習 (Residual RL)

学習速度と安定性を劇的に向上させるため、古典制御をベースとした「残差学習」へ移行する。

### 1. コンセプト
- **基本行動**: Pure Pursuit (純追従制御) がレーシングラインに基づき計算。
- **RLの役割**: ベース行動に対する「補正値（Residual）」のみを学習。
- **メリット**: ステップ 0 から完走可能。RLはラップタイム最適化に特化できる。

### 2. 実装項目
- [x] `src/controllers/pure_pursuit.py`: ベース制御器の実装
- [x] `src/config.py`: `USE_RESIDUAL_RL` およびスケール設定の追加
- [x] `src/f1_env.py`: `step()` 内で `final_action = base_action + residual` を計算
- [x] `src/rewards.py`: ベースラインに対する向上を評価する報酬の調整

---

## ■ 次のアクション

- [ ] 最新報酬系での学習開始 (scripts/train.py)
- [ ] 走行テスト (scripts/enjoy_wide.py) による Residual RL の挙動確認
- [ ] TensorBoard で `explained_variance` と `reward` の推移を監視