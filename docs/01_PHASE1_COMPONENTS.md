# Phase 1: 基礎コンポーネント実装ガイド

本フェーズでは、以下の3つの基礎コンポーネントを実装します：

1. **Model Health Monitor** - 推論の健全性監視
2. **Safety Manager** - 状態遷移・フェイルセーフ管理
3. **Collision Recovery** - 衝突復帰シーケンス

---

## 📌 実装順序

```
1. model_health.py      (単独で動作可能)
2. collision_recovery.py (Pure Pursuit Controller に依存)
3. safety.py            (1, 2 に依存) ← 最後に
```

---

## 1. ModelHealthMonitor (`src/model_health.py`)

### 目的

推論実行時間・出力値を監視し、以下を検出：
- **タイムアウト**: 推論が 20ms を超過
- **NaN/Inf**: 出力に無限値や NaN を含む
- **範囲外**: 出力が [-1.0, 1.0] 外に出ている

### 実装概要

```python
class ModelHealthMonitor:
    def __init__(self, timeout_sec: float = 0.020):
        self.timeout_sec = timeout_sec
        self.last_valid_action = None
        self.inference_times = deque(maxlen=10)  # 直近10フレーム
        self.start_time = None
    
    def start_inference(self):
        """推論開始時刻を記録"""
        self.start_time = time.time()
    
    def end_inference(self, action: np.ndarray) -> tuple[bool, dict]:
        """
        推論完了・検証
        Returns:
            (is_healthy, error_dict)
        """
        elapsed = time.time() - self.start_time
        self.inference_times.append(elapsed)
        
        errors = {}
        if elapsed > self.timeout_sec:
            errors['timeout'] = True
        if np.isnan(action).any():
            errors['nan'] = True
        if np.isinf(action).any():
            errors['inf'] = True
        if (action < -1.0).any() or (action > 1.0).any():
            errors['out_of_bounds'] = True
        
        if not errors:
            self.last_valid_action = action.copy()
            return True, {}
        else:
            return False, errors
    
    def get_avg_inference_time(self) -> float:
        return np.mean(self.inference_times) if self.inference_times else 0.0
    
    def reset(self):
        self.inference_times.clear()
        self.last_valid_action = None
```

### テスト方法

```python
# シミュレーション内でテスト
monitor = ModelHealthMonitor(timeout_sec=0.020)

# 正常な推論
monitor.start_inference()
time.sleep(0.010)  # 10ms
action = np.array([0.1, 0.5])
healthy, errors = monitor.end_inference(action)
assert healthy == True
assert errors == {}

# タイムアウト
monitor.start_inference()
time.sleep(0.025)  # 25ms > 20ms
action = np.array([0.1, 0.5])
healthy, errors = monitor.end_inference(action)
assert healthy == False
assert errors.get('timeout') == True

# NaN 検出
monitor.start_inference()
time.sleep(0.010)
action = np.array([np.nan, 0.5])
healthy, errors = monitor.end_inference(action)
assert healthy == False
assert errors.get('nan') == True
```

---

## 2. CollisionRecovery (`src/collision_recovery.py`)

### 目的

衝突後、段階的に復帰するシーケンスを実行：

```
[BACKING]    (1.5秒) ─→ [TURNING]      (2.0秒) ─→ [MOVING_FORWARD] (1.0秒)
speed=-0.3             steering=±0.4              speed=0.2 + PP
steering=0             speed=0
```

### 実装概要

```python
class CollisionRecoverySequence:
    """
    衝突復帰シーケンス管理
    状態遷移: IDLE → BACKING → TURNING → MOVING_FORWARD → COMPLETED
    """
    
    def __init__(self, pure_pursuit_controller, wheelbase: float = 0.33):
        self.pp_controller = pure_pursuit_controller
        self.wheelbase = wheelbase
        
        # 時間設定
        self.BACKING_TIME = 1.5
        self.TURNING_TIME = 2.0
        self.MOVING_FORWARD_TIME = 1.0
        
        # 状態
        self.state = "IDLE"
        self.start_time = None
        self.phase_start_time = None
        self.recovery_start_pose = None
    
    def start_recovery(self, x: float, y: float, yaw: float):
        """復帰シーケンス開始"""
        self.state = "BACKING"
        self.start_time = time.time()
        self.phase_start_time = self.start_time
        self.recovery_start_pose = (x, y, yaw)
    
    def compute_recovery_action(
        self,
        x: float,
        y: float,
        yaw: float,
        current_time: float
    ) -> Optional[tuple[float, float]]:
        """
        復帰シーケンス内のアクション計算
        Returns:
            (steering, speed) or None if completed
        """
        if self.state == "IDLE":
            return None
        
        phase_elapsed = current_time - self.phase_start_time
        
        # ========== BACKING ==========
        if self.state == "BACKING":
            if phase_elapsed < self.BACKING_TIME:
                # 直線後退
                return (0.0, -0.3)
            else:
                # 次フェーズへ
                self.state = "TURNING"
                self.phase_start_time = current_time
        
        # ========== TURNING ==========
        if self.state == "TURNING":
            if phase_elapsed < self.TURNING_TIME:
                # 目標方向を判定して回転
                start_yaw = self.recovery_start_pose[2]
                target_steer = 0.4 if (start_yaw % (2 * np.pi)) < np.pi else -0.4
                return (target_steer, 0.0)
            else:
                # 次フェーズへ
                self.state = "MOVING_FORWARD"
                self.phase_start_time = current_time
        
        # ========== MOVING_FORWARD ==========
        if self.state == "MOVING_FORWARD":
            if phase_elapsed < self.MOVING_FORWARD_TIME:
                # Pure Pursuit + 速度制限
                pp_action = self.pp_controller.compute_action(
                    (x, y, yaw, 0.0)  # 仮の速度 (使わない)
                )
                # 速度を制限 (0.2m/s 以下)
                return (pp_action[0], min(pp_action[1], 0.2))
            else:
                # 復帰完了
                self.state = "COMPLETED"
                return None
        
        return None
    
    def is_completed(self) -> bool:
        return self.state == "COMPLETED"
    
    def reset(self):
        self.state = "IDLE"
        self.start_time = None
        self.phase_start_time = None
        self.recovery_start_pose = None
```

### テスト方法

```python
# Mock Pure Pursuit Controller
class MockPurePursuit:
    def compute_action(self, robot_state):
        return (0.1, 0.5)  # dummy output

recovery = CollisionRecoverySequence(MockPurePursuit())
current_time = 0.0

# 復帰開始
recovery.start_recovery(x=0.0, y=0.0, yaw=0.0)
assert recovery.state == "BACKING"

# BACKING フェーズ
action = recovery.compute_recovery_action(0.0, 0.0, 0.0, current_time + 0.5)
assert action == (0.0, -0.3)

# TURNING フェーズへ移行
action = recovery.compute_recovery_action(0.0, 0.0, 0.0, current_time + 2.0)
assert recovery.state == "TURNING"
assert action == (-0.4, 0.0) or action == (0.4, 0.0)

# MOVING_FORWARD フェーズへ移行
action = recovery.compute_recovery_action(0.0, 0.0, 0.0, current_time + 4.5)
assert recovery.state == "MOVING_FORWARD"
assert action[1] <= 0.2  # 速度制限確認

# 復帰完了
action = recovery.compute_recovery_action(0.0, 0.0, 0.0, current_time + 6.0)
assert action is None
assert recovery.is_completed()
```

---

## 3. SafetyManager (`src/safety.py`)

### 目的

以下を管理：
- **状態遷移**: NORMAL → FALLBACK/COLLISION → SAFE_STOP
- **フォールバック戦略**: モデル失敗時は Pure Pursuit へ
- **段階的降速**: フォールバック中も速度を徐々に落とす

### 実装概要

```python
from enum import Enum

class SafetyState(Enum):
    NORMAL = 1              # モデル制御
    FALLBACK_TO_PP = 2      # Pure Pursuit フォールバック
    COLLISION_DETECTED = 3  # 衝突復帰実行中
    RECOVERY_BACKING = 4
    RECOVERY_TURNING = 5
    RECOVERY_MOVING = 6
    SAFE_STOP = 7           # 停止状態

class SafetyManager:
    """
    安全管理マネージャー
    状態遷移と段階的降速を管理
    """
    
    def __init__(
        self,
        model_monitor: ModelHealthMonitor,
        pure_pursuit_controller,
        collision_recovery: CollisionRecoverySequence,
        fallback_timeout: float = 10.0,
        collision_recovery_timeout: float = 5.0
    ):
        self.state = SafetyState.NORMAL
        self.model_monitor = model_monitor
        self.pp_controller = pure_pursuit_controller
        self.collision_recovery = collision_recovery
        
        # タイムアウト設定
        self.fallback_timeout = fallback_timeout
        self.collision_recovery_timeout = collision_recovery_timeout
        self.fallback_start_time = None
        self.collision_start_time = None
        
        # 段階的降速スケジュール
        self.speed_schedule = [1.0, 0.8, 0.5, 0.2, 0.0]
        self.speed_stage = 0
        self.speed_stage_duration = 2.0  # 各段階 2秒間
        self.last_speed_stage_time = None
    
    def update_state(
        self,
        model_healthy: bool,
        collision: bool,
        x: float,
        y: float,
        yaw: float,
        current_time: float
    ):
        """状態遷移を更新"""
        
        # NORMAL → 他の状態への遷移
        if self.state == SafetyState.NORMAL:
            if collision:
                self.state = SafetyState.COLLISION_DETECTED
                self.collision_start_time = current_time
                self.collision_recovery.start_recovery(x, y, yaw)
            elif not model_healthy:
                self.state = SafetyState.FALLBACK_TO_PP
                self.fallback_start_time = current_time
                self.speed_stage = 0
                self.last_speed_stage_time = current_time
        
        # FALLBACK_TO_PP 内での処理
        elif self.state == SafetyState.FALLBACK_TO_PP:
            elapsed = current_time - self.fallback_start_time
            
            if model_healthy:
                # モデルが復帰した
                self.state = SafetyState.NORMAL
                self.speed_stage = 0
            elif elapsed > self.fallback_timeout:
                # タイムアウト: 停止
                self.state = SafetyState.SAFE_STOP
            
            # 段階的降速の進行
            stage_elapsed = current_time - self.last_speed_stage_time
            if stage_elapsed > self.speed_stage_duration:
                if self.speed_stage < len(self.speed_schedule) - 1:
                    self.speed_stage += 1
                    self.last_speed_stage_time = current_time
        
        # COLLISION_DETECTED 内での処理
        elif self.state == SafetyState.COLLISION_DETECTED:
            elapsed = current_time - self.collision_start_time
            
            if self.collision_recovery.is_completed():
                # 復帰完了 → NORMAL へ戻る
                self.state = SafetyState.NORMAL
            elif elapsed > self.collision_recovery_timeout:
                # タイムアウト: 停止
                self.state = SafetyState.SAFE_STOP
    
    def get_action(
        self,
        model_action: Optional[np.ndarray],
        robot_state: Optional[tuple],
        x: float,
        y: float,
        yaw: float,
        current_time: float
    ) -> np.ndarray:
        """
        現在の状態に応じたアクション出力
        
        Args:
            model_action: NN モデルからの出力 [steering, speed]
            robot_state: (x, y, yaw, speed) など
            x, y, yaw: 現在位置・姿勢
            current_time: 現在時刻
        
        Returns:
            (steering, speed)
        """
        
        if self.state == SafetyState.NORMAL:
            return model_action if model_action is not None else np.array([0.0, 0.0])
        
        elif self.state == SafetyState.FALLBACK_TO_PP:
            # Pure Pursuit 制御 + 段階的降速
            pp_action = self.pp_controller.compute_action(robot_state)
            speed_factor = self.speed_schedule[min(self.speed_stage, len(self.speed_schedule) - 1)]
            return np.array([pp_action[0], pp_action[1] * speed_factor], dtype=np.float32)
        
        elif self.state == SafetyState.COLLISION_DETECTED:
            # 衝突復帰シーケンス
            recovery_action = self.collision_recovery.compute_recovery_action(x, y, yaw, current_time)
            if recovery_action is not None:
                return np.array(recovery_action, dtype=np.float32)
            else:
                # 復帰完了待ちの間
                return np.array([0.0, 0.0], dtype=np.float32)
        
        elif self.state == SafetyState.SAFE_STOP:
            # 完全停止
            return np.array([0.0, 0.0], dtype=np.float32)
        
        return np.array([0.0, 0.0], dtype=np.float32)
    
    def reset(self):
        """エピソード終了時のリセット"""
        self.state = SafetyState.NORMAL
        self.fallback_start_time = None
        self.collision_start_time = None
        self.speed_stage = 0
        self.last_speed_stage_time = None
        self.model_monitor.reset()
        self.collision_recovery.reset()
```

### テスト方法

```python
# Mock コンポーネント
class MockModelHealth:
    def __init__(self):
        self.healthy = True
    def reset(self):
        pass

class MockPurePursuit:
    def compute_action(self, state):
        return (0.1, 0.5)

class MockCollisionRecovery:
    def __init__(self):
        self.state = "IDLE"
    def start_recovery(self, x, y, yaw):
        self.state = "BACKING"
    def compute_recovery_action(self, x, y, yaw, t):
        return (0.0, -0.3) if self.state == "BACKING" else None
    def is_completed(self):
        return self.state == "COMPLETED"
    def reset(self):
        self.state = "IDLE"

# Safety Manager テスト
model_monitor = MockModelHealth()
pp = MockPurePursuit()
recovery = MockCollisionRecovery()
safety = SafetyManager(model_monitor, pp, recovery, fallback_timeout=2.0)

# NORMAL 状態でのアクション出力
model_action = np.array([0.1, 0.5])
action = safety.get_action(model_action, None, 0, 0, 0, 0)
assert safety.state == SafetyState.NORMAL
assert np.allclose(action, model_action)

# モデル失敗でフォールバック
model_monitor.healthy = False
safety.update_state(False, False, 0, 0, 0, 0)
assert safety.state == SafetyState.FALLBACK_TO_PP

# PP アクション取得
action = safety.get_action(None, (0, 0, 0, 0), 0, 0, 0, 0)
assert action[1] == 0.5  # 初期段階は速度 100%

# 段階的降速（2秒×5段階 = 10秒）
for stage in range(1, 5):
    safety.update_state(False, False, 0, 0, 0, stage * 2.0 + 0.1)
    action = safety.get_action(None, (0, 0, 0, 0), 0, 0, 0, stage * 2.0 + 0.1)
    expected_speed = 0.5 * safety.speed_schedule[stage]
    assert action[1] == expected_speed

# タイムアウトで停止
safety.update_state(False, False, 0, 0, 0, 15.0)
assert safety.state == SafetyState.SAFE_STOP

print("✓ All SafetyManager tests passed")
```

---

## ✅ Phase 1 完了チェックリスト

- [ ] `src/model_health.py` 実装完了
- [ ] `src/collision_recovery.py` 実装完了
- [ ] `src/safety.py` 実装完了
- [ ] 各コンポーネントのユニットテスト合格
- [ ] インテグレーションテスト（3つのコンポーネント併用）で合格
- [ ] コード品質チェック (PEP8, type hints)

---

## 次のステップ

→ `docs/02_PHASE2_SIMULATION.md` で `src/f1_env.py` へ統合
