#!/usr/bin/env python3
"""
record_demo.py — F1Tenth 実機デモ録画スクリプト
=====================================================
実機 F1Tenth (native ROS1 / ROS2) 上で動作。
人間がジョイスティックで走行中に LiDAR + 操舵/速度を同期録画し .npz で保存する。

【使い方】
  ROS2: python3 record_demo.py
  ROS1: python3 record_demo.py --ros1

【保存フォーマット】
  scans    : (N, M)   float32  生 LiDAR [m]  (M = ビーム数)
  steers   : (N,)     float32  ステアリング角 [rad]
  speeds   : (N,)     float32  速度 [m/s]
  stamps   : (N,)     float64  タイムスタンプ [s]

Ctrl+C で録画停止 → demo_YYYYMMDD_HHMMSS.npz として保存。
"""

import sys
import os
import time
import signal
import argparse
import threading
import numpy as np
from datetime import datetime
from collections import deque

# ─── 設定 ──────────────────────────────────────────────────
SCAN_TOPIC     = "/scan"                       # LiDARトピック
DRIVE_TOPIC    = "/drive"                      # ROS2 AckermannDriveStamped
DRIVE_TOPIC_R1 = "/ackermann_cmd"              # ROS1 AckermannDriveStamped (別名)
SAVE_DIR       = os.path.expanduser("~/demos") # 保存先ディレクトリ
SYNC_TOL_SEC   = 0.05                          # タイムスタンプ同期許容差 [s]
# ────────────────────────────────────────────────────────────


class DemoRecorder:
    """LiDAR と Drive コマンドを同期録画するクラス"""

    def __init__(self):
        self.scans_buf: deque  = deque(maxlen=5000)
        self.drives_buf: deque = deque(maxlen=5000)

        self.recorded_scans  = []
        self.recorded_steers = []
        self.recorded_speeds = []
        self.recorded_stamps = []

        self.lock = threading.Lock()
        self.running = True

    # ──── ROS2 ─────────────────────────────────────────────

    def run_ros2(self):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import LaserScan
        from ackermann_msgs.msg import AckermannDriveStamped

        rclpy.init()
        node = rclpy.create_node("demo_recorder")

        def scan_cb(msg: LaserScan):
            t = rclpy.clock.Clock().now().nanoseconds * 1e-9
            ranges = np.array(msg.ranges, dtype=np.float32)
            ranges = np.nan_to_num(ranges, nan=30.0, posinf=30.0, neginf=0.0)
            with self.lock:
                self.scans_buf.append((t, ranges))

        def drive_cb(msg: AckermannDriveStamped):
            t   = rclpy.clock.Clock().now().nanoseconds * 1e-9
            st  = float(msg.drive.steering_angle)
            spd = float(msg.drive.speed)
            with self.lock:
                self.drives_buf.append((t, st, spd))
                self._try_pair()

        node.create_subscription(LaserScan, SCAN_TOPIC, scan_cb, 10)
        node.create_subscription(AckermannDriveStamped, DRIVE_TOPIC, drive_cb, 10)

        print(f"[ROS2] 録画開始: {SCAN_TOPIC} + {DRIVE_TOPIC}")
        print("  Ctrl+C で停止 → .npz 保存")

        while rclpy.ok() and self.running:
            rclpy.spin_once(node, timeout_sec=0.1)

        node.destroy_node()
        rclpy.shutdown()

    # ──── ROS1 ─────────────────────────────────────────────

    def run_ros1(self):
        import rospy
        from sensor_msgs.msg import LaserScan
        from ackermann_msgs.msg import AckermannDriveStamped

        rospy.init_node("demo_recorder", anonymous=True)

        def scan_cb(msg: LaserScan):
            t = msg.header.stamp.to_sec()
            ranges = np.array(msg.ranges, dtype=np.float32)
            ranges = np.nan_to_num(ranges, nan=30.0, posinf=30.0, neginf=0.0)
            with self.lock:
                self.scans_buf.append((t, ranges))

        def drive_cb(msg: AckermannDriveStamped):
            t   = msg.header.stamp.to_sec()
            st  = float(msg.drive.steering_angle)
            spd = float(msg.drive.speed)
            with self.lock:
                self.drives_buf.append((t, st, spd))
                self._try_pair()

        # どちらのトピック名でも受け取れるように両方購読
        rospy.Subscriber(SCAN_TOPIC,     LaserScan,             scan_cb)
        rospy.Subscriber(DRIVE_TOPIC,    AckermannDriveStamped, drive_cb)
        rospy.Subscriber(DRIVE_TOPIC_R1, AckermannDriveStamped, drive_cb)

        print(f"[ROS1] 録画開始: {SCAN_TOPIC} + {DRIVE_TOPIC}/{DRIVE_TOPIC_R1}")
        print("  Ctrl+C で停止 → .npz 保存")
        rospy.spin()

    # ──── 同期ペアリング ────────────────────────────────────

    def _try_pair(self):
        """最新の Drive メッセージに最近傍の Scan を対応付けて保存"""
        if not self.scans_buf or not self.drives_buf:
            return

        t_drv, st, spd = self.drives_buf[-1]

        # scans_buf の中で t_drv に最も近いものを探す
        best_t, best_scan = None, None
        for t_s, scan in self.scans_buf:
            if best_t is None or abs(t_s - t_drv) < abs(best_t - t_drv):
                best_t, best_scan = t_s, scan

        if best_t is None:
            return
        if abs(best_t - t_drv) > SYNC_TOL_SEC:
            return  # 時刻差が大きすぎるペアは捨てる

        self.recorded_scans.append(best_scan)
        self.recorded_steers.append(st)
        self.recorded_speeds.append(spd)
        self.recorded_stamps.append(t_drv)

        # 進捗表示
        n = len(self.recorded_stamps)
        if n % 100 == 0:
            print(f"  ペア記録: {n} steps  |  速度: {spd:.2f} m/s  "
                  f"ステア: {st:.3f} rad")

    # ──── 保存 ──────────────────────────────────────────────

    def save(self):
        if not self.recorded_stamps:
            print("[WARN] 保存データが 0 件です。")
            return

        os.makedirs(SAVE_DIR, exist_ok=True)
        fname = datetime.now().strftime("demo_%Y%m%d_%H%M%S.npz")
        path  = os.path.join(SAVE_DIR, fname)

        np.savez_compressed(
            path,
            scans  = np.array(self.recorded_scans,  dtype=np.float32),
            steers = np.array(self.recorded_steers, dtype=np.float32),
            speeds = np.array(self.recorded_speeds, dtype=np.float32),
            stamps = np.array(self.recorded_stamps, dtype=np.float64),
        )
        n = len(self.recorded_stamps)
        dur = self.recorded_stamps[-1] - self.recorded_stamps[0]
        print(f"\n✅ 保存完了: {path}")
        print(f"   ステップ数: {n}  /  録画時間: {dur:.1f} s  /  "
              f"平均 {n/dur:.1f} Hz")


def main():
    global SAVE_DIR, DRIVE_TOPIC, DRIVE_TOPIC_R1
    parser = argparse.ArgumentParser(description="F1Tenth 実機デモ録画")
    parser.add_argument("--ros1", action="store_true",
                        help="ROS1 (rospy) モードで起動 (デフォルト: ROS2)")
    parser.add_argument("--drive-topic", type=str, default=None,
                        help="Drive トピック名を上書き")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help=f"保存先ディレクトリ (default: {SAVE_DIR})")
    args = parser.parse_args()

    SAVE_DIR = args.save_dir
    if args.drive_topic:
        DRIVE_TOPIC    = args.drive_topic
        DRIVE_TOPIC_R1 = args.drive_topic



    recorder = DemoRecorder()

    def on_sigint(sig, frame):
        print("\n[STOP] 録画を停止します...")
        recorder.running = False
        recorder.save()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)

    print("=" * 50)
    print("  F1Tenth Demo Recorder")
    print(f"  保存先: {SAVE_DIR}")
    if args.ros1:
        print("  モード: ROS1 (rospy)")
    else:
        print("  モード: ROS2 (rclpy)")
    print("  Ctrl+C で停止 → demo_YYYYMMDD_HHMMSS.npz 保存")
    print("=" * 50)

    try:
        if args.ros1:
            recorder.run_ros1()
        else:
            recorder.run_ros2()
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        recorder.save()


if __name__ == "__main__":
    main()
