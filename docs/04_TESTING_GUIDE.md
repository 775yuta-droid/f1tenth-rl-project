# テスト実施ガイド

本ガイドではシミュレーション環境と Jetson 実機環境でのテスト方法を説明します。

---

## 🔄 テスト順序

```
1. Phase 1 単体テスト (コンポーネント各 3-4 時間)
2. Phase 2 シミュレーション統合テスト (4 時間)
3. Phase 3 Jetson 環境テスト (2-3 時間)
4. 本番走行テスト (別途スケジュール)
```

---

## Phase 1: コンポーネント単体テスト

### 1.1 ModelHealthMonitor テスト

```bash
cd /home/toyot/projects/f1tenth-rl-project

# テストスクリプト
python -c "
from src.model_health import ModelHealthMonitor
import numpy as np
import time

monitor = ModelHealthMonitor(timeout_sec=0.020)

# Test 1: 正常な推論
print('Test 1: Normal inference')
monitor.start_inference()
time.sleep(0.010)
action = np.array([0.1, 0.5])
healthy, errors = monitor.end_inference(action)
assert healthy == True, 'Expected healthy'
print('✓ Passed')

# Test 2: タイムアウト
print('Test 2: Timeout')
monitor.start_inference()
time.sleep(0.025)
action = np.array([0.1, 0.5])
healthy, errors = monitor.end_inference(action)
assert healthy == False, 'Expected unhealthy'
assert 'timeout' in errors, 'Expected timeout error'
print('✓ Passed')

# Test 3: NaN 検出
print('Test 3: NaN detection')
monitor.start_inference()
time.sleep(0.010)
action = np.array([np.nan, 0.5])
healthy, errors = monitor.end_inference(action)
assert healthy == False, 'Expected unhealthy'
assert 'nan' in errors, 'Expected NaN error'
print('✓ Passed')

print('\\nAll ModelHealthMonitor tests passed ✓')
"
```

### 1.2 SafetyManager テスト

```bash
python -c "
from src.safety import SafetyManager, SafetyState
from src.model_health import ModelHealthMonitor
import numpy as np

# Mock コンポーネント
class MockPP:
    def compute_action(self, state):
        return (0.1, 0.5)

class MockCollisionRecovery:
    def __init__(self):
        self.state = 'IDLE'
    def start_recovery(self, x, y, yaw):
        self.state = 'RECOVERING'
    def compute_recovery_action(self, x, y, yaw, t):
        return (0.0, -0.3) if self.state == 'RECOVERING' else None
    def is_completed(self):
        return False
    def reset(self):
        self.state = 'IDLE'

model_monitor = ModelHealthMonitor()
pp = MockPP()
recovery = MockCollisionRecovery()
safety = SafetyManager(model_monitor, pp, recovery)

# Test: NORMAL → FALLBACK
print('Test: State transition NORMAL → FALLBACK')
assert safety.state == SafetyState.NORMAL
safety.update_state(False, False, 0, 0, 0, 0)
assert safety.state == SafetyState.FALLBACK_TO_PP
print('✓ Passed')

print('\\nAll SafetyManager tests passed ✓')
"
```

---

## Phase 2: シミュレーション統合テスト

### 2.1 テスト実行

```bash
cd /home/toyot/projects/f1tenth-rl-project

# 統合テストスクリプト実行
python scripts/test_safety_system.py
```

### 2.2 期待される出力

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

### 2.3 結果の確認ポイント

| 項目 | 期待値 | 検証方法 |
|:---:|:---:|:---|
| **正常走行** | 完走率 > 90% | Test 1 の 100 ステップ中に done に至るかを確認 |
| **PP フォールバック** | 無限ループしない | Test 2 で 200 ステップ以内に FALLBACK 検知 |
| **衝突復帰** | 少なくとも 1 回の復帰 | Test 3 で 500 ステップ内に衝突が発生して復帰シーケンス開始 |
| **段階的降速** | 10秒以内に停止 | Test 4 で FALLBACK → SAFE_STOP の遷移時間が 3-4 秒 |

---

## Phase 3: Jetson 環境テスト

### 3.1 Jetson 環境セットアップ確認

```bash
# Jetson で実行
cd /home/toyot/projects/f1tenth-rl-project

# 環境確認
python -c "
import onnxruntime as ort
print('ONNX Runtime available:', hasattr(ort, 'InferenceSession'))

import rclpy
print('ROS2 available:', hasattr(rclpy, 'init'))

import numpy as np
print('NumPy version:', np.__version__)
"
```

### 3.2 ONNX 推論速度テスト

```bash
python -c "
import onnxruntime as ort
import numpy as np
import time

print('Testing ONNX inference speed...')

# モデルロード
session = ort.InferenceSession(
    'models/best_model.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

input_name = session.get_inputs()[0].name
dummy_input = np.random.randn(1, 864).astype(np.float32)

# 30回の推論時間を測定
times = []
for i in range(30):
    start = time.perf_counter()
    session.run(None, {input_name: dummy_input})
    times.append((time.perf_counter() - start) * 1000)  # ms に変換

avg_time = np.mean(times)
max_time = np.max(times)
min_time = np.min(times)

print(f'Inference time: {avg_time:.2f}ms (min: {min_time:.2f}ms, max: {max_time:.2f}ms)')

if avg_time < 18:
    print('✓ Inference speed acceptable (< 18ms)')
else:
    print('⚠ Warning: Inference speed may be slow')
"
```

### 3.3 ROS2 ノード実行テスト

```bash
# Terminal 1: Policy Node 起動
ros2 run f1tenth_rl jetson_policy_node

# Terminal 2: Safety Node 起動
ros2 run f1tenth_rl ros2_safety_node

# Terminal 3: 監視スクリプト実行
python sharing/test_jetson_integration.py
```

### 3.4 トピック通信確認

```bash
# Terminal A: Policy Node トピック確認
ros2 topic list
ros2 topic echo /action

# Terminal B: Safety Node トピック確認
ros2 topic echo /safe_action
ros2 topic echo /model_health
```

期待される出力：
```
publisher: /jetson_policy_node
  subscriber: /ros2_safety_node

---
[0.15, 0.45]
[0.15, 0.47]
[0.16, 0.46]
...
```

### 3.5 ダミーセンサデータ送信テスト

```python
# dummy_scan_publisher.py
import rclpy
from sensor_msgs.msg import LaserScan
import numpy as np

def main():
    rclpy.init()
    node = rclpy.create_node('dummy_scan_publisher')
    pub = node.create_publisher(LaserScan, '/scan', 10)
    
    # ダミースキャンデータ
    scan_msg = LaserScan()
    scan_msg.header.frame_id = 'laser'
    scan_msg.angle_min = -2.35  # -135°
    scan_msg.angle_max = 2.35   # +135°
    scan_msg.angle_increment = 0.00436  # 1440ビーム
    scan_msg.ranges = [1.0] * 1440  # 1m の距離
    
    for i in range(100):
        scan_msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(scan_msg)
        node.get_logger().info(f'Published scan {i}')
        rclpy.spin_once(node, timeout_sec=0.05)
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

```bash
python dummy_scan_publisher.py
```

---

## 本番走行前の最終チェック

### ✅ シミュレーション側

- [ ] Phase 1-3 のテスト全て PASS
- [ ] 安全設定 (`config.py`) が正しく読み込まれている
- [ ] ログファイルが `logs/safety_events/` に出力されている

### ✅ Jetson 側

- [ ] ONNX 推論速度が 18ms 以下
- [ ] ROS2 ノード起動時にエラーが出ていない
- [ ] `/action`, `/safe_action`, `/model_health` トピックが通信できている
- [ ] 緊急停止トピック (`/emergency_stop`) が機能している
- [ ] 衝突復帰に使用する `my_maps/my_map.csv` が存在する

### ✅ ハードウェア側

- [ ] Jetson の温度が正常範囲 (< 70°C)
- [ ] LiDAR が正常に動作している (スキャン頻度 40Hz)
- [ ] VESC モーター制御が正常に応答している
- [ ] バッテリー電圧が安定している

---

## トラブルシューティング

### ONNX 推論が遅い場合

```bash
# CUDAExecutionProvider が使用されているか確認
python -c "
import onnxruntime as ort
session = ort.InferenceSession('models/best_model.onnx')
print('Providers:', session.get_providers())
"
```

期待: `['CUDAExecutionProvider', 'CPUExecutionProvider']`

**対策**: CUDA ドライバ・cuDNN を確認

### ROS2 トピックが通信されない場合

```bash
# ノード間の接続確認
ros2 node list
ros2 node info /jetson_policy_node
ros2 node info /ros2_safety_node

# トピック確認
ros2 topic list
ros2 topic info /action
```

**対策**: ファイアウォール設定・ネットワーク接続を確認

### モデルエラー「無効な入力形状」

```python
# 入力形状確認
import onnxruntime as ort
session = ort.InferenceSession('models/best_model.onnx')
for inp in session.get_inputs():
    print(f"Input: {inp.name}, Shape: {inp.shape}")
```

**対策**: LiDAR スタック方法がモデルの入力形状と一致しているか確認

---

## 次のステップ

→ [本番走行](06_TROUBLESHOOTING.md) に向けてのチェックリスト確認
