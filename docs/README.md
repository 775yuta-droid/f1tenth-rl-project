# Jetson 実装プロジェクト - ドキュメント目次

本ディレクトリには、モデル失敗時対策および衝突復帰システムの実装ガイドが含まれています。

---

## 📚 ドキュメント一覧

| # | ファイル | 内容 | 対象者 |
|:---:|:---:|:---|:---|
| **00** | [00_IMPLEMENTATION_OVERVIEW.md](00_IMPLEMENTATION_OVERVIEW.md) | 全体システム構成・フェーズ概要・クイックスタート | 全員 (必読) |
| **01** | [01_PHASE1_COMPONENTS.md](01_PHASE1_COMPONENTS.md) | ModelHealthMonitor, SafetyManager, CollisionRecovery の実装詳細 | デベロッパー |
| **02** | [02_PHASE2_SIMULATION.md](02_PHASE2_SIMULATION.md) | `src/f1_env.py`, `config.py` への統合とシミュレーション側テスト | デベロッパー |
| **03** | [03_PHASE3_JETSON_ROS2.md](03_PHASE3_JETSON_ROS2.md) | Jetson 上の ROS2 ノード実装・ONNX 推論・安全管理 | Jetson 担当 |
| **04** | [04_TESTING_GUIDE.md](04_TESTING_GUIDE.md) | Phase 1-3 テスト手順・検証方法・期待値 | テスター |
| **05** | [05_JETSON_SETUP.md](05_JETSON_SETUP.md) | Jetson 環境セットアップ手順・パッケージインストール・設定 | Jetson 管理者 |
| **06** | [06_TROUBLESHOOTING.md](06_TROUBLESHOOTING.md) | 実機走行時のトラブルシューティング・FAQ | オペレータ |

---

## 🚀 実装の流れ

```
Step 1: 00_IMPLEMENTATION_OVERVIEW を読む (全体像把握)
   ↓
Step 2: 01_PHASE1_COMPONENTS でコンポーネント実装 (2-3日)
   ↓
Step 3: 02_PHASE2_SIMULATION でシミュレーション統合 (1日)
   ↓
Step 4: 04_TESTING_GUIDE でテスト実施 (テスト内容確認)
   ↓
Step 5: 05_JETSON_SETUP で Jetson 環境構築 (1-2日)
   ↓
Step 6: 03_PHASE3_JETSON_ROS2 で ROS2 ノード実装 (3-4日)
   ↓
Step 7: 04_TESTING_GUIDE で実機テスト (2-3日)
   ↓
Step 8: 06_TROUBLESHOOTING でトラブル対応 (本番走行時)
```

---

## 📋 ファイル構成

```
docs/
├── 00_IMPLEMENTATION_OVERVIEW.md  ← 最初にここから!
├── 01_PHASE1_COMPONENTS.md        ← コンポーネント実装
├── 02_PHASE2_SIMULATION.md        ← シミュレーション統合
├── 03_PHASE3_JETSON_ROS2.md       ← Jetson ROS2 実装
├── 04_TESTING_GUIDE.md            ← テスト手順
├── 05_JETSON_SETUP.md             ← 環境セットアップ
└── 06_TROUBLESHOOTING.md          ← トラブルシューティング

src/
├── safety.py                      ← 新規: 状態管理
├── model_health.py                ← 新規: 推論監視
├── collision_recovery.py           ← 新規: 衝突復帰
├── f1_env.py                      ← 修正: Safety 統合
└── config.py                      ← 修正: Safety 設定追加

sharing/
├── jetson_policy_node.py          ← 新規: ONNX 推論ノード
├── ros2_safety_node.py            ← 新規: 安全管理ノード
├── jetson_main.py                 ← 新規: 統合実行スクリプト
└── test_jetson_integration.py      ← 新規: Jetson テスト

scripts/
└── test_safety_system.py          ← 新規: シミュレーション テスト
```

---

## ⚡ クイックレファレンス

### 主要パラメータ

| パラメータ | デフォルト | ファイル | 調整理由 |
|:---:|:---:|:---|:---|
| `MODEL_INFERENCE_TIMEOUT_SEC` | 0.020 | config.py | Jetson 性能に応じて |
| `FALLBACK_PP_TIMEOUT_SEC` | 10.0 | config.py | モデル復帰待機時間 |
| `FALLBACK_SPEED_SCHEDULE` | [1.0, 0.8, 0.5, 0.2, 0.0] | config.py | 段階的降速スケジュール |
| `COLLISION_BACKING_TIME_SEC` | 1.5 | collision_recovery.py | コース幅に応じて |
| `COLLISION_TURNING_TIME_SEC` | 2.0 | collision_recovery.py | 回転必要時間 |

### 状態遷移

```
NORMAL
  ├─→ model_healthy=False ──→ FALLBACK_TO_PP
  │      └─→ (10秒超) ──→ SAFE_STOP
  │      └─→ recovery ──→ NORMAL
  │
  └─→ collision=True ──→ COLLISION_DETECTED
       ├─→ BACKING ──→ TURNING ──→ MOVING_FORWARD ──→ NORMAL
       └─→ (5秒超) ──→ SAFE_STOP

SAFE_STOP (terminal state)
```

---

## 🔗 関連リソース

- **レーシングライン生成**: `scripts/utils/generate_centerline.py`
- **Pure Pursuit コントローラ**: `src/controllers/pure_pursuit.py`
- **報酬計算**: `src/rewards.py`
- **設定管理**: `src/config.py`

---

## 📞 サポート

各ドキュメントの末尾に「次のステップ」が記載されています。  
進捗に応じて適切なドキュメントを参照してください。

---

## ✅ 実装進捗チェック

```
Phase 1 (コンポーネント実装)
  ☐ ModelHealthMonitor
  ☐ SafetyManager
  ☐ CollisionRecovery

Phase 2 (シミュレーション統合)
  ☐ src/f1_env.py 修正
  ☐ src/config.py 設定追加
  ☐ scripts/test_safety_system.py 実装

Phase 3 (Jetson/ROS2 実装)
  ☐ jetson_policy_node.py
  ☐ ros2_safety_node.py
  ☐ jetson_main.py

Phase 4 (テスト)
  ☐ シミュレーション テスト (4シナリオ)
  ☐ Jetson テスト (ONNX速度, ROS2通信)
  ☐ 実機走行テスト

Phase 5 (ドキュメント & デプロイ)
  ☐ 本番走行マニュアル作成
  ☐ Jetson パッケージ化
  ☐ 運用ガイド整備
```

最初のドキュメントは **00_IMPLEMENTATION_OVERVIEW.md** から始めてください。

Happy coding! 🚀
