# Jetson セットアップガイド

本番走行前に Jetson 環境を整備するためのステップバイステップガイドです。

---

## 前提条件

- Jetson Nano / Xavier / Orin がセットアップ済み
- Ubuntu 20.04 / 22.04 がインストール済み
- ROS2 Foxy / Humble がセットアップ済み

---

## 📋 セットアップ手順

### Step 1: 基本パッケージのインストール

```bash
sudo apt-get update
sudo apt-get upgrade -y

# Python 3.8 以上が必要
sudo apt-get install -y python3-pip python3-dev

# ONNX Runtime インストール
# (GPU サポート)
pip3 install onnxruntime-gpu

# または (CPU のみ)
pip3 install onnxruntime

# ROS2 関連
sudo apt-get install -y python3-colcon-common-extensions
pip3 install rclpy
```

### Step 2: プロジェクトコードの配置

```bash
# Jetson 上の作業ディレクトリ
mkdir -p ~/f1tenth_ws
cd ~/f1tenth_ws

# プロジェクトのコピー
# (GitHub から clone または scp で転送)
git clone <your-repo-url>
cd f1tenth-rl-project

# 仮想環境作成 (オプション)
python3 -m venv venv
source venv/bin/activate

# 依存関係インストール
pip3 install -r requirements.txt
```

### Step 3: ONNX モデルの配置

```bash
# モデルファイルが存在することを確認
ls -la models/

# 出力例:
# -rw-r--r-- 1 user user 12345678 May 22 12:34 best_model.onnx
# -rw-r--r-- 1 user user 23456789 May 22 12:34 ppo_model_exp39.onnx
```

### Step 4: レーシングラインデータの配置

```bash
# コース情報 (CSV ファイル)
ls -la my_maps/

# 出力例:
# -rw-r--r-- 1 user user 98765   May 22 12:34 my_map.csv
# -rw-r--r-- 1 user user 123456  May 22 12:34 my_map.pgm
# -rw-r--r-- 1 user user 567     May 22 12:34 my_map.yaml
```

### Step 5: ROS2 ワークスペース構築

```bash
cd ~/f1tenth_ws

# ワークスペース初期化 (必要に応じて)
colcon build --symlink-install

# または単にパスに追加
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Step 6: 環境変数設定

```bash
# ~/.bashrc に追加
cat >> ~/.bashrc << 'EOF'

# F1Tenth ROS2 Settings
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Jetson F1Tenth Project
export PYTHONPATH=$PYTHONPATH:$HOME/f1tenth_ws/f1tenth-rl-project
export MODEL_PATH=$HOME/f1tenth_ws/f1tenth-rl-project/models/best_model.onnx
export RACING_LINE_PATH=$HOME/f1tenth_ws/f1tenth-rl-project/my_maps/my_map.csv
EOF

source ~/.bashrc
```

### Step 7: 接続の確認

```bash
# LiDAR デバイスの確認
ls -la /dev/ttyUSB*
# または
ros2 topic list | grep scan

# VESC モーター制御の確認
ros2 topic list | grep /drive
```

### Step 8: 動作テスト

```bash
# ノード起動テスト
python3 sharing/jetson_main.py

# 別ターミナルでトピック監視
ros2 topic list
ros2 topic echo /action  # モデル出力を確認
```

---

## 🔧 詳細設定

### ONNX Runtime の GPU サポート確認

```bash
python3 -c "
import onnxruntime as ort

print('Available Execution Providers:')
for provider in ort.get_available_providers():
    print(f'  - {provider}')

# GPU が利用可能な場合
if 'CUDAExecutionProvider' in ort.get_available_providers():
    print('✓ GPU support enabled')
else:
    print('⚠ GPU support disabled (using CPU)')
"
```

**結果の例:**
```
Available Execution Providers:
  - CUDAExecutionProvider
  - CPUExecutionProvider
✓ GPU support enabled
```

### 電力管理 (Jetson Nano の場合)

```bash
# Jetson Nano では電力消費が大きいため、モード設定が重要
sudo jetson_clocks --show
sudo nvpmodel -m 0  # 最大パフォーマンスモード
```

### 熱対策

```bash
# 温度監視
watch -n 1 nvidia-smi

# 目安:
# - < 60°C: 安全
# - 60-70°C: 警告
# - > 70°C: 危険 (スロットル開始)
```

---

## 📝 設定ファイル例

### `/etc/systemd/system/f1tenth-policy.service`

自動起動サービス（オプション）:

```ini
[Unit]
Description=F1Tenth Policy Executor
After=network-online.target ros2.service

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/f1tenth_ws/f1tenth-rl-project
ExecStart=/usr/bin/python3 sharing/jetson_main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

起動/停止:
```bash
sudo systemctl start f1tenth-policy
sudo systemctl status f1tenth-policy
sudo systemctl stop f1tenth-policy

# ログ確認
sudo journalctl -u f1tenth-policy -f
```

---

## 🚨 緊急停止ボタン設定

ROS2 から緊急停止トピック (`/emergency_stop`) を発行するシンプルなスクリプト:

```python
# emergency_stop_publisher.py
import rclpy
from std_msgs.msg import Bool
import sys

def main():
    rclpy.init()
    node = rclpy.create_node('emergency_stop_publisher')
    pub = node.create_publisher(Bool, '/emergency_stop', 10)
    
    # 確認メッセージ
    print("Emergency Stop Publisher")
    print("Press Enter to publish emergency stop...")
    input()
    
    msg = Bool(data=True)
    pub.publish(msg)
    print("✓ Emergency stop published")
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

```bash
python3 emergency_stop_publisher.py
```

---

## ✅ セットアップチェックリスト

実機走行前に確認してください：

- [ ] ONNX Runtime が GPU サポート対応で動作
- [ ] ROS2 ノード (Policy + Safety) が起動可能
- [ ] `/scan` トピックが LiDAR から流れている
- [ ] `/action` トピックが推論出力を出力している
- [ ] `/safe_action` トピックが安全制御出力を出力している
- [ ] モデル推論時間が 18ms 以下
- [ ] Jetson 温度が 60°C 以下
- [ ] 緊急停止トピック (`/emergency_stop`) が機能している
- [ ] バッテリー電圧が安定している (表示値で確認)
- [ ] モーター制御 (VESC) が反応している

---

## 🐛 トラブルシューティング

### 問題 1: ONNX Runtime インストール失敗

```bash
# GPU なしでも動作するように
pip3 install onnxruntime

# CPU 版で動作確認後、GPU 版をインストール
pip3 install onnxruntime-gpu
```

### 問題 2: ROS2 トピック通信なし

```bash
# domain ID を確認
echo $ROS_DOMAIN_ID

# 同じ domain ID で起動
export ROS_DOMAIN_ID=0
python3 sharing/jetson_main.py
```

### 問題 3: メモリ不足エラー

```bash
# 利用可能メモリ確認
free -h

# プロセス監視
ps aux | grep python

# メモリ使用量が多い場合、モデルを軽量化するか、
# フレームスタック数を削減 (config.py の FRAME_STACK)
```

### 問題 4: モデルの入力形状エラー

```bash
# 入力形状を確認
python3 -c "
from src.config import config
from src.racing_line import RacingLine

print(f'LiDAR size: {1080 // config.LIDAR_DOWNSAMPLE_FACTOR}')
print(f'Racing line size: {RacingLine.NUM_FEATURES if config.INCLUDE_RACING_LINE else 0}')
print(f'Expected total: {1080 // config.LIDAR_DOWNSAMPLE_FACTOR + RacingLine.NUM_FEATURES}')

# ONNX モデルの入力形状
import onnxruntime as ort
session = ort.InferenceSession('models/best_model.onnx')
for inp in session.get_inputs():
    print(f'Model input shape: {inp.shape}')
"
```

---

## 次のステップ

→ [テスト実施](04_TESTING_GUIDE.md) に進む  
→ [トラブルシューティング](06_TROUBLESHOOTING.md) を参照
