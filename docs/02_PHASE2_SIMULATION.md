# Phase 2: シミュレーション環境への統合

Phase 1 で実装したコンポーネントを、シミュレーション環境 (`src/f1_env.py`) に組み込みます。

---

## 📌 変更ファイル

1. **`src/f1_env.py`** - Safety Manager 統合
2. **`src/config.py`** - Safety 関連設定追加

---

## 1. `src/config.py` への追加設定

### 追加内容

`src/config.py` の末尾に以下を追加：

```python
# ============================================================
# Safety & Failsafe Settings
# ============================================================

# Safety Manager の有効化
ENABLE_SAFETY_MANAGER = True

# 衝突復帰機能の有効化
ENABLE_COLLISION_RECOVERY = True

# モデル推論のタイムアウト [秒]
# 40Hz 制御 (25ms/フレーム) に対し、20ms でタイムアウト
MODEL_INFERENCE_TIMEOUT_SEC = 0.020

# フォールバック時の最大継続時間 [秒]
# この時間を超過するとシステムは安全停止に移行
FALLBACK_PP_TIMEOUT_SEC = 10.0

# 衝突復帰シーケンスの最大実行時間 [秒]
COLLISION_RECOVERY_TIMEOUT_SEC = 5.0

# 段階的降速スケジュール
# モデル失敗時、各段階 2秒間で速度を段階的に低減
# [100%, 80%, 50%, 20%, 0%]
FALLBACK_SPEED_SCHEDULE = [1.0, 0.8, 0.5, 0.2, 0.0]

# 降速の段階継続時間 [秒]
FALLBACK_SPEED_STAGE_DURATION_SEC = 2.0

# ============================================================
# Collision Recovery Sequence Parameters
# ============================================================

# バック走行時間 [秒]
COLLISION_BACKING_TIME_SEC = 1.5

# ステアリング回転時間 [秒]
COLLISION_TURNING_TIME_SEC = 2.0

# 復帰後の前進時間 [秒]
COLLISION_MOVING_FORWARD_TIME_SEC = 1.0

# 復帰時の速度制限 [m/s]
COLLISION_RECOVERY_MAX_SPEED = 0.2
```

---

## 2. `src/f1_env.py` への統合

### 2.1 初期化 (`__init__` メソッド)

`F1TenthRL.__init__` の最後に以下を追加：

```python
# ============================================================
# Safety Components Initialization
# ============================================================

if config.ENABLE_SAFETY_MANAGER:
    from .model_health import ModelHealthMonitor
    from .safety import SafetyManager
    from .collision_recovery import CollisionRecoverySequence
    
    self.model_monitor = ModelHealthMonitor(
        timeout_sec=config.MODEL_INFERENCE_TIMEOUT_SEC
    )
    
    self.collision_recovery = CollisionRecoverySequence(
        self.pp_controller,
        wheelbase=config.CAR_LENGTH * 0.7
    ) if config.ENABLE_COLLISION_RECOVERY else None
    
    self.safety_manager = SafetyManager(
        self.model_monitor,
        self.pp_controller,
        self.collision_recovery,
        fallback_timeout=config.FALLBACK_PP_TIMEOUT_SEC,
        collision_recovery_timeout=config.COLLISION_RECOVERY_TIMEOUT_SEC
    )
else:
    self.model_monitor = None
    self.safety_manager = None
    self.collision_recovery = None
```

### 2.2 Step メソッドの修正

`F1TenthRL.step()` メソッドを以下のように修正：

**変更前:**
```python
def step(self, action):
    # ... 既存コード ...
    obs, reward, done, info = self.env.step(...)
    return obs, reward, done, info
```

**変更後:**
```python
def step(self, action):
    """
    Step の実行
    
    フロー:
    1. モデル出力検証
    2. 衝突検知
    3. 状態更新
    4. アクション決定（安全性を考慮）
    5. シミュレータ実行
    """
    import time
    
    # ========================================
    # A. モデル推論結果の検証
    # ========================================
    if self.safety_manager:
        self.model_monitor.start_inference()
        model_healthy, error_info = self.model_monitor.end_inference(action)
    else:
        model_healthy = True
        error_info = {}
    
    # ========================================
    # B. 現在の観測と状態を取得
    # ========================================
    raw_obs = self.env.unwrapped.obs_dict(self.env.unwrapped.agents[0])
    x = raw_obs['poses_x'][0]
    y = raw_obs['poses_y'][0]
    yaw = raw_obs['poses_theta'][0]
    speed = raw_obs['linear_vels_x'][0]
    steering = raw_obs['steering_angle'][0]
    
    # ========================================
    # C. 安全管理による状態遷移と処理
    # ========================================
    current_time = time.time()
    
    if self.safety_manager:
        # 衝突検知（一度の step では collision がセットされる）
        # 次のステップで検知可能な状態にするため、前フレームの info を参照
        collision = getattr(self, '_last_collision', False)
        
        # 状態更新
        self.safety_manager.update_state(
            model_healthy,
            collision,
            x, y, yaw,
            current_time
        )
        
        # アクション決定
        robot_state = (x, y, yaw, speed)
        safe_action = self.safety_manager.get_action(
            action,
            robot_state,
            x, y, yaw,
            current_time
        )
        
        action_to_use = safe_action
    else:
        action_to_use = action
    
    # ========================================
    # D. シミュレータステップ実行
    # ========================================
    obs, reward, done, info = self.env.step(action_to_use)
    
    # ========================================
    # E. 衝突フラグの記録（次ステップ用）
    # ========================================
    collision_current = info.get('collision', False)
    self._last_collision = collision_current
    
    # ========================================
    # F. 安全情報をログに含める
    # ========================================
    if self.safety_manager:
        info['safety_state'] = self.safety_manager.state.name
        info['model_healthy'] = model_healthy
        info['model_error_info'] = error_info
        info['model_avg_inference_time'] = self.model_monitor.get_avg_inference_time()
    
    return obs, reward, done, info
```

### 2.3 Reset メソッドの修正

`F1TenthRL.reset()` メソッドにセーフティ初期化を追加：

```python
def reset(self):
    """
    エピソードリセット
    Safety Manager も同時にリセット
    """
    obs = self.env.reset()
    
    if self.safety_manager:
        self.safety_manager.reset()
    
    # 衝突フラグの初期化
    self._last_collision = False
    
    return obs
```

---

## 3. `src/f1_env.py` への追加メソッド（オプション）

### 状態取得ヘルパー

```python
def get_safety_state(self) -> dict:
    """
    現在のセーフティ状態を取得
    
    Returns:
        {
            'state': SafetyState.name,
            'model_healthy': bool,
            'speed_stage': int,
            'collision_recovery_active': bool
        }
    """
    if not self.safety_manager:
        return {}
    
    return {
        'state': self.safety_manager.state.name,
        'model_healthy': self.model_monitor.last_valid_action is not None,
        'speed_stage': self.safety_manager.speed_stage,
        'collision_recovery_active': (
            self.safety_manager.state.name.startswith('RECOVERY')
        ),
        'avg_inference_time': self.model_monitor.get_avg_inference_time()
    }
```

### ログ記録ヘルパー

```python
def log_safety_event(self, event_type: str, details: dict = None):
    """
    セーフティイベントをログファイルに記録
    
    Args:
        event_type: 'MODEL_TIMEOUT', 'COLLISION_DETECTED', 'FALLBACK_START' など
        details: イベント詳細
    """
    import json
    from pathlib import Path
    
    log_dir = Path('logs/safety_events')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'safety_log_{timestamp}.jsonl'
    
    event = {
        'timestamp': time.time(),
        'event_type': event_type,
        'safety_state': self.safety_manager.state.name if self.safety_manager else 'N/A',
        'details': details or {}
    }
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(event) + '\n')
```

---

## 4. テスト: `scripts/test_safety_system.py`

Phase 2 統合のテストを実施します。

### 4.1 基本的なテストスケルトン

```python
import numpy as np
import sys
from pathlib import Path

# パスを調整
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.f1_env import F1TenthRL
import time

def test_normal_operation():
    """通常動作: モデル推論が正常な場合"""
    print("Test 1: Normal Operation")
    env = F1TenthRL('my_maps/my_map')
    
    obs = env.reset()
    
    for step in range(100):
        # 正常なアクション
        action = np.array([0.1, 0.5], dtype=np.float32)
        obs, reward, done, info = env.step(action)
        
        # 検証
        assert info['safety_state'] == 'NORMAL', \
            f"Expected NORMAL, got {info['safety_state']}"
        assert info['model_healthy'] == True
        
        if done:
            break
    
    print("✓ Test 1 passed\n")

def test_model_timeout():
    """モデルタイムアウト時の PP フォールバック"""
    print("Test 2: Model Timeout → Fallback to Pure Pursuit")
    env = F1TenthRL('my_maps/my_map')
    obs = env.reset()
    
    timeout_detected = False
    fallback_detected = False
    
    for step in range(200):
        if step < 50:
            # 正常なアクション
            action = np.array([0.1, 0.5], dtype=np.float32)
        else:
            # タイムアウトを発生させる
            action = np.array([np.nan, 0.5], dtype=np.float32)
        
        obs, reward, done, info = env.step(action)
        
        # フォールバック検知
        if step > 50 and info['safety_state'] == 'FALLBACK_TO_PP':
            fallback_detected = True
            break
        
        if done:
            break
    
    assert fallback_detected, "Fallback was not triggered"
    print("✓ Test 2 passed\n")

def test_collision_recovery():
    """衝突復帰シーケンスの実行"""
    print("Test 3: Collision Recovery Sequence")
    env = F1TenthRL('my_maps/my_map')
    obs = env.reset()
    
    collision_detected = False
    recovery_started = False
    
    for step in range(500):
        action = np.array([0.5, 0.8], dtype=np.float32)  # 高速で走行
        obs, reward, done, info = env.step(action)
        
        # 衝突検知
        if info.get('collision', False):
            collision_detected = True
        
        # 復帰シーケンス検知
        if collision_detected and 'RECOVERY' in info['safety_state']:
            recovery_started = True
            print(f"  Recovery phase: {info['safety_state']}")
        
        if done:
            break
    
    if collision_detected:
        assert recovery_started, "Recovery sequence was not triggered"
        print("✓ Test 3 passed\n")
    else:
        print("⚠ Test 3 skipped: No collision in this run\n")

def test_fallback_timeout():
    """フォールバックタイムアウト時の停止"""
    print("Test 4: Fallback Timeout → Safe Stop")
    env = F1TenthRL('my_maps/my_map')
    
    # タイムアウト値を短く設定
    if env.safety_manager:
        env.safety_manager.fallback_timeout = 3.0
    
    obs = env.reset()
    start_time = time.time()
    safe_stop_reached = False
    
    for step in range(500):
        # 常にタイムアウトを発生
        action = np.array([np.nan, 0.5], dtype=np.float32)
        obs, reward, done, info = env.step(action)
        
        if info['safety_state'] == 'SAFE_STOP':
            safe_stop_reached = True
            elapsed = time.time() - start_time
            print(f"  Safe stop reached in {elapsed:.2f}s")
            break
        
        if done:
            break
    
    assert safe_stop_reached, "Safe stop was not reached"
    print("✓ Test 4 passed\n")

if __name__ == '__main__':
    print("=" * 60)
    print("Safety System Integration Tests")
    print("=" * 60 + "\n")
    
    try:
        test_normal_operation()
        test_model_timeout()
        test_collision_recovery()
        test_fallback_timeout()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
```

### 4.2 テスト実行

```bash
cd /home/toyot/projects/f1tenth-rl-project

# テストスクリプトを実行
python scripts/test_safety_system.py
```

期待出力：
```
============================================================
Safety System Integration Tests
============================================================

Test 1: Normal Operation
✓ Test 1 passed

Test 2: Model Timeout → Fallback to Pure Pursuit
✓ Test 2 passed

Test 3: Collision Recovery Sequence
  Recovery phase: COLLISION_DETECTED
  Recovery phase: RECOVERY_BACKING
  Recovery phase: RECOVERY_TURNING
  Recovery phase: RECOVERY_MOVING
✓ Test 3 passed

Test 4: Fallback Timeout → Safe Stop
  Safe stop reached in 3.12s
✓ Test 4 passed

============================================================
All tests passed! ✓
============================================================
```

---

## ✅ Phase 2 完了チェックリスト

- [ ] `src/config.py` に Safety 設定追加
- [ ] `src/f1_env.py` に Safety Manager 統合
- [ ] `src/f1_env.py` の `__init__`, `step`, `reset` を修正
- [ ] `scripts/test_safety_system.py` を実装
- [ ] 全テスト (test 1-4) 合格
- [ ] ログ出力で安全イベントが記録されることを確認

---

## 次のステップ

→ `docs/03_PHASE3_JETSON_ROS2.md` で Jetson/ROS2 実装に進む
