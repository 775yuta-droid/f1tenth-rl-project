# モデル失敗時対策 & 衝突復帰システム - 実装ガイド

## 🎯 全体目標

Jetson 実機走行でモデル推論が失敗・遅延した場合、**Pure Pursuit 単独での安全走行** へ自動切り替え。  
衝突時は **段階的復帰シーケンス**（バック→回転→前進）を実行する。

---

## 📊 システム構成図

```
┌─────────────────────────────────────────────────────────┐
│  Jetson ROS2 環境                                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  [LiDAR Driver]                                         │
│         ↓ /scan                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  jetson_policy_node.py                           │  │
│  │  - ONNX 推論実行                                 │  │
│  │  - 推論タイムアウト検出 (18ms)                   │  │
│  │  - ヘルスチェック監視                            │  │
│  │  → /action (推論出力)                            │  │
│  │  → /model_health (状態)                          │  │
│  └──────────────────────────────────────────────────┘  │
│         ↓                                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │  ros2_safety_node.py                             │  │
│  │  - 状態遷移管理                                  │  │
│  │  - モデル失敗 → PP フォールバック               │  │
│  │  - 衝突復帰シーケンス実行                        │  │
│  │  → /safe_action (確定アクション)                │  │
│  │  ← /emergency_stop (緊急停止)                    │  │
│  └──────────────────────────────────────────────────┘  │
│         ↓ /safe_action                                  │
│  [Motor Driver (VESC)]                                 │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 📋 実装フェーズ

| フェーズ | 内容 | 依存関係 | 予定期間 |
|:---:|:---:|:---:|:---:|
| **Phase 1** | シミュレーション側: Safety Manager, Model Health, Collision Recovery | なし | 2-3 日 |
| **Phase 2** | `src/f1_env.py` に Safety 統合 | Phase 1 | 1 日 |
| **Phase 3** | Jetson/ROS2 ノード実装 | Phase 1 | 3-4 日 |
| **Phase 4** | テスト & 検証 | Phase 2, 3 | 2 日 |
| **Phase 5** | ドキュメント & チューニング | 全フェーズ | 1 日 |

---

## 📁 ファイルツリー (実装後)

```
docs/
  ├─ 00_IMPLEMENTATION_OVERVIEW.md        (本ファイル)
  ├─ 01_PHASE1_COMPONENTS.md             (基礎コンポーネント設計)
  ├─ 02_PHASE2_SIMULATION.md             (シミュレーション統合)
  ├─ 03_PHASE3_JETSON_ROS2.md            (Jetson/ROS2 実装)
  ├─ 04_TESTING_GUIDE.md                 (テスト手順)
  ├─ 05_JETSON_SETUP.md                  (Jetson セットアップ)
  └─ 06_TROUBLESHOOTING.md               (トラブルシューティング)

src/
  ├─ safety.py                           (NEW: 状態管理)
  ├─ model_health.py                     (NEW: 推論監視)
  ├─ collision_recovery.py                (NEW: 衝突復帰)
  ├─ f1_env.py                           (MODIFIED: Safety 統合)
  └─ config.py                           (MODIFIED: Safety 設定)

sharing/
  ├─ jetson_policy_node.py               (NEW: ポリシー推論ノード)
  ├─ ros2_safety_node.py                 (NEW: 安全管理ノード)
  ├─ jetson_main.py                      (NEW: 統合エントリポイント)
  ├─ test_jetson_integration.py           (NEW: 実機テスト)
  └─ JETSON_DEPLOYMENT_PLAN.md           (UPDATED)

scripts/
  └─ test_safety_system.py               (NEW: シミュレーション テスト)
```

---

## 🔄 状態遷移フロー

```
START
  ↓
[NORMAL] ─────────────────────────────────────────┐
  ├─→ (model_healthy=False) ──→ [FALLBACK_TO_PP] │
  │                                    ├─→ (timeout > 10s) ──→ [SAFE_STOP]
  │                                    └─→ (recovery OK) ────→ [NORMAL]
  │                                                            ↑
  └─→ (collision=True) ──→ [COLLISION_DETECTED]             │
                              ├─→ [BACKING_UP]              │
                              ├─→ [TURNING]                 │
                              ├─→ [MOVING_FORWARD] ─────────┘
                              └─→ (timeout) ──→ [SAFE_STOP]

[SAFE_STOP] (terminal)
  └─→ speed=0, steering=0 (全状態から到達可能)
```

---

## 🚀 クイックスタート

### 1. 基礎コンポーネント実装 (Phase 1)

```bash
cd /home/toyot/projects/f1tenth-rl-project

# 以下のファイルを作成
# - src/model_health.py
# - src/safety.py
# - src/collision_recovery.py

# 詳細は docs/01_PHASE1_COMPONENTS.md を参照
```

### 2. シミュレーション統合 (Phase 2)

```bash
# src/f1_env.py に Safety Manager 組み込み
# src/config.py に Safety 設定追加
# 詳細は docs/02_PHASE2_SIMULATION.md を参照
```

### 3. Jetson/ROS2 実装 (Phase 3)

```bash
# Jetson 環境で以下を実装
# - sharing/jetson_policy_node.py
# - sharing/ros2_safety_node.py
# - sharing/jetson_main.py

# 詳細は docs/03_PHASE3_JETSON_ROS2.md を参照
```

### 4. テスト & 検証 (Phase 4)

```bash
# シミュレーション側
python scripts/test_safety_system.py

# Jetson 側（実機）
python sharing/test_jetson_integration.py
```

---

## ⚙️ 主要パラメータ

| パラメータ | 値 | 説明 |
|:---:|:---:|:---|
| `MODEL_INFERENCE_TIMEOUT_SEC` | 0.020 | 推論最大実行時間 (20ms, 40Hz制御想定) |
| `FALLBACK_PP_TIMEOUT_SEC` | 10.0 | フォールバック最大継続時間 |
| `COLLISION_RECOVERY_TIMEOUT_SEC` | 5.0 | 衝突復帰最大実行時間 |
| `FALLBACK_SPEED_SCHEDULE` | [1.0, 0.8, 0.5, 0.2, 0.0] | 段階的降速 (5段階) |
| `BACKING_TIME_SEC` | 1.5 | バック時間 |
| `TURNING_TIME_SEC` | 2.0 | 回転時間 |
| `MOVING_FORWARD_TIME_SEC` | 1.0 | 前進再開時間 |

詳細設定は各フェーズのドキュメントを参照。

---

## 📍 次のステップ

**今すぐ始める場合:**

1. `docs/01_PHASE1_COMPONENTS.md` を読む
2. `src/model_health.py` から実装を開始
3. 各コンポーネント完成後、`scripts/test_safety_system.py` で単体テスト
4. Phase 2 → Phase 3 と進める

**Jetson 実機作業に向けて:**

- `docs/05_JETSON_SETUP.md` で Jetson 環境を整備
- `docs/03_PHASE3_JETSON_ROS2.md` で実装開始
- `docs/04_TESTING_GUIDE.md` で本番テストを実施
