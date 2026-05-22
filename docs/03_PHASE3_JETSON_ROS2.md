# Phase 3: Jetson/ROS2 実装ガイド

Phase 2 まででシミュレーション側は完成。  
本フェーズでは Jetson 上で実機走行用の ROS2 ノードを実装します。

---

## 📌 主要実装ファイル

1. **`sharing/jetson_policy_node.py`** - ONNX 推論ノード
2. **`sharing/ros2_safety_node.py`** - 安全管理ノード
3. **`sharing/jetson_main.py`** - 統合実行スクリプト

---

## 1. Jetson Policy Node (`sharing/jetson_policy_node.py`)

### 目的

- Jetson 上で ONNX モデルを実行
- LiDAR データを受け取り、推論を実施
- 推論タイムアウト・エラー検出
- ヘルスチェック情報を発行

### 実装概要

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Float32
import onnxruntime as ort
import numpy as np
from collections import deque
import time
import os

class JetsonPolicyNode(Node):
    """
    Jetson 上の ONNX 推論ノード
    
    Subscribes:
        /scan: LiDAR スキャン (LaserScan)
    
    Publishes:
        /action: 推論出力アクション (Float32MultiArray)
        /model_health: ヘルスチェック情報 (Float32MultiArray)
        /inference_time: 推論実行時間 (Float32)
    """
    
    def __init__(self):
        super().__init__('jetson_policy_node')
        
        # ONNX モデルロード
        model_path = os.getenv('MODEL_PATH', 'models/best_model.onnx')
        self.get_logger().info(f"Loading ONNX model from: {model_path}")
        
        try:
            self.ort_session = ort.InferenceSession(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.get_logger().info("✓ Model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise
        
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        
        # タイムアウト設定
        self.timeout_sec = 0.018  # 18ms
        self.max_consecutive_failures = 5
        
        # 状態
        self.inference_times = deque(maxlen=20)
        self.failed_inferences = 0
        self.consecutive_failures = 0
        self.health_status = "HEALTHY"
        self.last_valid_action = np.array([0.0, 0.0], dtype=np.float32)
        
        # LiDAR バッファ（フレームスタック）
        self.lidar_buffer = deque(maxlen=4)
        self.lidar_scan_params = None
        
        # ROS2 インターフェース
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        self.action_pub = self.create_publisher(
            Float32MultiArray,
            '/action',
            10
        )
        
        self.health_pub = self.create_publisher(
            Float32MultiArray,
            '/model_health',
            10
        )
        
        self.inference_time_pub = self.create_publisher(
            Float32,
            '/inference_time',
            10
        )
        
        # 40Hz 推論タイマー (25ms 周期)
        self.timer = self.create_timer(0.025, self.inference_timer_callback)
        
        self.get_logger().info("JetsonPolicyNode initialized")
    
    def scan_callback(self, msg: LaserScan):
        """
        LiDAR スキャン受信・前処理
        
        処理内容:
        1. スキャン範囲を 270° に制限 (実機 Hokuyo URG)
        2. NaN/Inf クリーニング
        3. ダウンサンプリング (10 → 5)
        4. バッファに追加
        """
        # スキャン範囲を記録
        if self.lidar_scan_params is None:
            self.lidar_scan_params = {
                'angle_min': msg.angle_min,
                'angle_max': msg.angle_max,
                'angle_increment': msg.angle_increment,
            }
        
        try:
            # [180:1260] = 270°範囲の 1080 点
            scans_raw = np.array(msg.ranges[180:1260], dtype=np.float32)
            
            # クリーニング (NaN/Inf → 30.0)
            scans_clean = np.nan_to_num(
                scans_raw,
                nan=30.0,
                posinf=30.0,
                neginf=0.0
            )
            
            # クリップ [0, 30]
            scans_clipped = np.clip(scans_clean, 0.0, 30.0)
            
            # ダウンサンプリング: 1080 → 216点
            # (ダウンサンプリング係数=5: 1080/5=216)
            downsampled = scans_clipped.reshape(216, 5).min(axis=1)
            
            # バッファに追加
            self.lidar_buffer.append(downsampled)
            
        except Exception as e:
            self.get_logger().warn(f"Error processing LiDAR scan: {e}")
    
    def inference_timer_callback(self):
        """
        40Hz 推論タイマーコールバック
        
        フロー:
        1. フレームスタック確認
        2. ONNX 推論実行
        3. 結果検証
        4. パブリッシュ
        """
        # フレームスタック確認 (4フレーム = 100ms)
        if len(self.lidar_buffer) < 4:
            return
        
        try:
            # ========================================
            # 1. 入力準備: フレームスタック作成
            # ========================================
            stacked = np.concatenate(
                list(self.lidar_buffer),
                axis=0
            ).reshape(1, -1).astype(np.float32)
            
            # ========================================
            # 2. 推論実行（タイムアウト保護）
            # ========================================
            start_time = time.perf_counter()
            outputs = self.ort_session.run(
                None,
                {self.input_name: stacked}
            )
            elapsed = time.perf_counter() - start_time
            
            # 推論時間ログ
            self.inference_times.append(elapsed)
            
            # ========================================
            # 3. 出力検証
            # ========================================
            action = outputs[0][0].astype(np.float32)
            is_valid = self._validate_action(action, elapsed)
            
            if is_valid:
                self.last_valid_action = action.copy()
                self.consecutive_failures = 0
                self.health_status = "HEALTHY"
            else:
                self.consecutive_failures += 1
                self.failed_inferences += 1
                
                if self.consecutive_failures >= self.max_consecutive_failures:
                    self.health_status = "UNHEALTHY"
                
                # 前回の正常な値を再利用
                action = self.last_valid_action
            
            # ========================================
            # 4. パブリッシュ
            # ========================================
            self._publish_action(action)
            self._publish_health()
            self._publish_inference_time(elapsed)
            
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            self.health_status = "ERROR"
            self.consecutive_failures += 1
            self._publish_action(self.last_valid_action)
            self._publish_health()
    
    def _validate_action(self, action: np.ndarray, inference_time: float) -> bool:
        """アクション検証"""
        
        # タイムアウト判定
        if inference_time > self.timeout_sec:
            self.get_logger().warn(
                f"Inference timeout: {inference_time*1000:.2f}ms > {self.timeout_sec*1000:.2f}ms"
            )
            return False
        
        # NaN/Inf 判定
        if np.isnan(action).any():
            self.get_logger().warn("Action contains NaN")
            return False
        if np.isinf(action).any():
            self.get_logger().warn("Action contains Inf")
            return False
        
        # 範囲判定 [-1, 1]
        if (action < -1.0).any() or (action > 1.0).any():
            self.get_logger().warn(f"Action out of bounds: {action}")
            return False
        
        return True
    
    def _publish_action(self, action: np.ndarray):
        """推論アクション発行"""
        msg = Float32MultiArray()
        msg.data = action.tolist()
        self.action_pub.publish(msg)
    
    def _publish_health(self):
        """ヘルスチェック情報発行"""
        msg = Float32MultiArray()
        avg_inference_time = (
            np.mean(self.inference_times)
            if self.inference_times
            else 0.0
        )
        msg.data = [
            1.0 if self.health_status == "HEALTHY" else 0.0,
            avg_inference_time,
            float(self.consecutive_failures),
            float(self.failed_inferences)
        ]
        self.health_pub.publish(msg)
    
    def _publish_inference_time(self, elapsed: float):
        """推論実行時間発行"""
        msg = Float32(data=float(elapsed))
        self.inference_time_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = JetsonPolicyNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 2. ROS2 Safety Node (`sharing/ros2_safety_node.py`)

### 目的

- Policy Node からのアクションを監視
- ヘルスチェック情報を受け取り、状態遷移を判定
- Pure Pursuit フォールバック・衝突復帰を実行
- 最終的なアクションを `/safe_action` で発行

### 実装概要

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
import numpy as np
from collections import deque
import time
import os
import sys

# 一つ上のディレクトリから import
sys.path.insert(0, os.path.dirname(__file__))

class ROS2SafetyNode(Node):
    """
    Jetson 上の安全管理ノード
    
    Subscribes:
        /action: ONNX モデル出力 (Float32MultiArray)
        /model_health: ヘルスチェック情報 (Float32MultiArray)
        /emergency_stop: 緊急停止フラグ (Bool)
        /odometry/filtered: ロボット姿勢 (nav_msgs/Odometry)
    
    Publishes:
        /safe_action: 確定アクション (Float32MultiArray)
        /safety_state: 安全状態 (Float32MultiArray)
    """
    
    def __init__(self):
        super().__init__('ros2_safety_node')
        
        # 状態管理
        self.state = "NORMAL"  # NORMAL, FALLBACK, COLLISION_RECOVERY, SAFE_STOP
        self.model_healthy = True
        self.collision_detected = False
        self.fallback_start_time = None
        self.recovery_start_time = None
        self.recovery_phase = None
        
        # タイムアウト設定
        self.fallback_timeout = 10.0
        self.collision_recovery_timeout = 5.0
        
        # Pure Pursuit 初期化
        self._init_pure_pursuit()
        
        # 衝突復帰初期化
        self._init_collision_recovery()
        
        # バッファ
        self.last_model_action = np.array([0.0, 0.0], dtype=np.float32)
        self.last_valid_action = np.array([0.0, 0.0], dtype=np.float32)
        
        # 段階的降速
        self.speed_schedule = [1.0, 0.8, 0.5, 0.2, 0.0]
        self.speed_stage = 0
        self.speed_stage_time = None
        
        # 現在の位置・姿勢
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        # ROS2 インターフェース
        self.action_sub = self.create_subscription(
            Float32MultiArray,
            '/action',
            self.action_callback,
            10
        )
        
        self.health_sub = self.create_subscription(
            Float32MultiArray,
            '/model_health',
            self.health_callback,
            10
        )
        
        self.emergency_sub = self.create_subscription(
            Bool,
            '/emergency_stop',
            self.emergency_callback,
            10
        )
        
        self.safe_action_pub = self.create_publisher(
            Float32MultiArray,
            '/safe_action',
            10
        )
        
        self.safety_state_pub = self.create_publisher(
            Float32MultiArray,
            '/safety_state',
            10
        )
        
        # 監視タイマー (10Hz)
        self.timer = self.create_timer(0.1, self.safety_timer_callback)
        
        self.get_logger().info("ROS2SafetyNode initialized")
    
    def _init_pure_pursuit(self):
        """Pure Pursuit コントローラ初期化"""
        try:
            from src.controllers.pure_pursuit import PurePursuitController
            from src.racing_line import RacingLine
            
            # レーシングライン読み込み
            racing_line_path = os.getenv(
                'RACING_LINE_PATH',
                'my_maps/my_map.csv'
            )
            
            self.racing_line = RacingLine(racing_line_path)
            
            # Pure Pursuit コントローラ
            self.pp_controller = PurePursuitController(
                self.racing_line,
                wheelbase=0.33,
                lookahead_dist=0.6
            )
            
            self.get_logger().info("✓ Pure Pursuit initialized")
        except Exception as e:
            self.get_logger().warn(f"Pure Pursuit init failed: {e}")
            self.pp_controller = None
    
    def _init_collision_recovery(self):
        """衝突復帰シーケンス初期化"""
        try:
            from src.collision_recovery import CollisionRecoverySequence
            
            self.collision_recovery = CollisionRecoverySequence(self.pp_controller)
            self.get_logger().info("✓ Collision recovery initialized")
        except Exception as e:
            self.get_logger().warn(f"Collision recovery init failed: {e}")
            self.collision_recovery = None
    
    def action_callback(self, msg: Float32MultiArray):
        """モデル出力アクション受信"""
        self.last_model_action = np.array(msg.data, dtype=np.float32)
    
    def health_callback(self, msg: Float32MultiArray):
        """モデルヘルスチェック受信"""
        health_ok = msg.data[0] > 0.5
        avg_inference_time = msg.data[1]
        consecutive_failures = int(msg.data[2])
        
        self.model_healthy = health_ok
        
        if health_ok:
            self.last_valid_action = self.last_model_action
        
        # ログ
        if consecutive_failures > 0:
            self.get_logger().info(
                f"Model health: {'OK' if health_ok else 'NG'}, "
                f"avg_time={avg_inference_time*1000:.2f}ms, "
                f"consecutive_failures={consecutive_failures}"
            )
    
    def emergency_callback(self, msg: Bool):
        """緊急停止トピック受信"""
        if msg.data:
            self.state = "SAFE_STOP"
            self.get_logger().error("Emergency stop activated!")
    
    def safety_timer_callback(self):
        """安全管理タイマー (10Hz)"""
        current_time = time.time()
        
        # ========================================
        # 状態遷移
        # ========================================
        self._update_state(current_time)
        
        # ========================================
        # アクション決定
        # ========================================
        safe_action = self._compute_action(current_time)
        
        # ========================================
        # パブリッシュ
        # ========================================
        self._publish_safe_action(safe_action)
        self._publish_safety_state(current_time)
    
    def _update_state(self, current_time: float):
        """状態遷移更新"""
        
        # NORMAL → 他の状態へ
        if self.state == "NORMAL":
            if not self.model_healthy:
                self.state = "FALLBACK"
                self.fallback_start_time = current_time
                self.speed_stage = 0
                self.speed_stage_time = current_time
                self.get_logger().info("State → FALLBACK")
        
        # FALLBACK 内での処理
        elif self.state == "FALLBACK":
            elapsed = current_time - self.fallback_start_time
            
            if self.model_healthy:
                # モデルが復帰
                self.state = "NORMAL"
                self.speed_stage = 0
                self.get_logger().info("State → NORMAL (model recovered)")
            
            elif elapsed > self.fallback_timeout:
                # タイムアウト
                self.state = "SAFE_STOP"
                self.get_logger().error("State → SAFE_STOP (fallback timeout)")
            
            # 段階的降速の進行 (2秒ごと)
            stage_elapsed = current_time - self.speed_stage_time
            if stage_elapsed > 2.0:
                if self.speed_stage < len(self.speed_schedule) - 1:
                    self.speed_stage += 1
                    self.speed_stage_time = current_time
                    self.get_logger().info(
                        f"Speed stage → {self.speed_stage} "
                        f"({self.speed_schedule[self.speed_stage]*100:.0f}%)"
                    )
    
    def _compute_action(self, current_time: float) -> np.ndarray:
        """状態に応じたアクション計算"""
        
        if self.state == "NORMAL":
            return self.last_model_action
        
        elif self.state == "FALLBACK":
            # Pure Pursuit + 段階的降速
            if self.pp_controller:
                try:
                    pp_action = self.pp_controller.compute_action(
                        (self.current_x, self.current_y, self.current_yaw, 0.0)
                    )
                    speed_factor = self.speed_schedule[
                        min(self.speed_stage, len(self.speed_schedule) - 1)
                    ]
                    return np.array(
                        [pp_action[0], pp_action[1] * speed_factor],
                        dtype=np.float32
                    )
                except Exception as e:
                    self.get_logger().warn(f"PP computation failed: {e}")
                    return np.array([0.0, 0.2], dtype=np.float32)  # Safe default
            else:
                return self.last_valid_action
        
        elif self.state == "SAFE_STOP":
            return np.array([0.0, 0.0], dtype=np.float32)
        
        return np.array([0.0, 0.0], dtype=np.float32)
    
    def _publish_safe_action(self, action: np.ndarray):
        """確定アクション発行"""
        msg = Float32MultiArray()
        msg.data = action.tolist()
        self.safe_action_pub.publish(msg)
    
    def _publish_safety_state(self, current_time: float):
        """安全状態発行"""
        msg = Float32MultiArray()
        msg.data = [
            1.0 if self.state == "NORMAL" else 0.0,
            float(self.model_healthy),
            float(self.speed_stage),
        ]
        self.safety_state_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ROS2SafetyNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 3. Jetson 統合メインプログラム (`sharing/jetson_main.py`)

### 目的

両ノードを一括で起動・管理

```python
#!/usr/bin/env python3
"""
Jetson メイン実行スクリプト

使用方法:
    python jetson_main.py

環境変数:
    MODEL_PATH: ONNX モデルのパス
    RACING_LINE_PATH: レーシングラインの CSV ファイルパス
"""

import subprocess
import signal
import sys
import os
import time

def run_nodes():
    """ROS2 ノードを起動"""
    
    # 環境変数設定
    os.environ.setdefault('MODEL_PATH', 'models/best_model.onnx')
    os.environ.setdefault('RACING_LINE_PATH', 'my_maps/my_map.csv')
    
    print("=" * 60)
    print("Jetson F1Tenth Policy Executor")
    print("=" * 60)
    print(f"Model: {os.environ['MODEL_PATH']}")
    print(f"Racing line: {os.environ['RACING_LINE_PATH']}")
    print("=" * 60 + "\n")
    
    # ノードプロセス管理
    processes = []
    
    try:
        # Policy Node 起動
        print("[1/2] Starting Jetson Policy Node...")
        policy_proc = subprocess.Popen(
            [sys.executable, '-m', 'sharing.jetson_policy_node'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(('jetson_policy_node', policy_proc))
        time.sleep(1)  # 起動待機
        
        # Safety Node 起動
        print("[2/2] Starting ROS2 Safety Node...")
        safety_proc = subprocess.Popen(
            [sys.executable, '-m', 'sharing.ros2_safety_node'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(('ros2_safety_node', safety_proc))
        
        print("\n✓ All nodes started successfully\n")
        print("Press Ctrl+C to stop...")
        
        # ノード実行監視
        while True:
            for name, proc in processes:
                if proc.poll() is not None:
                    # プロセスが終了した
                    print(f"\n⚠ {name} terminated unexpectedly")
                    raise RuntimeError(f"{name} crashed")
            
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    
    finally:
        # クリーンアップ
        for name, proc in processes:
            if proc.poll() is None:
                print(f"Stopping {name}...")
                proc.terminate()
                proc.wait(timeout=5)
        
        print("✓ All nodes stopped")

if __name__ == '__main__':
    run_nodes()
```

---

## ✅ Phase 3 完了チェックリスト

- [ ] `sharing/jetson_policy_node.py` 実装完了
- [ ] `sharing/ros2_safety_node.py` 実装完了
- [ ] `sharing/jetson_main.py` 実装完了
- [ ] ONNX モデルを Jetson で実行可能か確認 (< 18ms)
- [ ] ROS2 トピックの通信が正常か確認
- [ ] Pure Pursuit コントローラが Jetson で動作可能か確認

---

## 次のステップ

→ `docs/04_TESTING_GUIDE.md` でテスト手順を確認  
→ `docs/05_JETSON_SETUP.md` で Jetson 環境セットアップを参照
