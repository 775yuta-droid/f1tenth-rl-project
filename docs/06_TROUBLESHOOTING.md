# トラブルシューティング & FAQ

本番走行中に遭遇する可能性がある問題と対処法をまとめています。

---

## 実機走行時のよくある問題

### 🔴 モデル推論タイムアウト

**症状:**  
- ログに「Inference timeout: 25.3ms > 20ms」と出力
- `/model_health` トピックで `health=0`
- 車が Pure Pursuit に自動切り替わる

**原因:**
1. Jetson 温度が高い (> 70°C)
2. 他のプロセスが CPU/GPU リソースを消費している
3. ONNX Runtime が CPU フォールバックしている

**対処:**
```bash
# 1. 温度確認
nvidia-smi
watch -n 1 nvidia-smi  # リアルタイム監視

# 2. プロセス確認
top
ps aux | sort -k3 -r | head -20

# 3. GPU 状態確認
nvidia-smi -pm 1
nvidia-smi -pm 0

# 4. ONNX Runtime の確認
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('Providers:', providers)
# CUDAExecutionProvider が最初に来ていればOK
"
```

**解決策:**
- Jetson の冷却ファンを増強
- 背景プロセスを終了
- CPU/GPU クロック周波数を確認
- 必要に応じてモデルを軽量化

---

### 🔴 衝突復帰の失敗 (無限ループ)

**症状:**
- 衝突後、BACKING → TURNING → MOVING_FORWARD のシーケンスが無限ループ
- 車が同じ場所で回転し続ける

**原因:**
1. レーシングラインのデータが不正確
2. 位置推定 (odometry) がずれている
3. 復帰シーケンスのタイミング設定が適切でない

**対処:**
```python
# collision_recovery.py のタイムアウト値を短くする
BACKING_TIME = 1.0         # 1.5秒 → 1.0秒
TURNING_TIME = 1.5         # 2.0秒 → 1.5秒
MOVING_FORWARD_TIME = 0.8  # 1.0秒 → 0.8秒

# または、復帰シーケンス開始前に停止時間を入れる
if collision:
    self.state = "STOP_AND_ASSESS"
    # 0.5秒停止して状態を確認
    time.sleep(0.5)
```

**確認方法:**
```bash
# ログから復帰シーケンスの詳細を確認
tail -f logs/safety_events/safety_log_*.jsonl | grep RECOVERY

# 位置推定の精度を確認
ros2 topic echo /odometry/filtered
```

---

### 🔴 Pure Pursuit フォールバック後の無限降速

**症状:**
- モデル失敗後、速度が 1.0 → 0.8 → 0.5 → 0.2 → 0.0 と段階的に低下
- 10秒後に完全停止して復帰しない

**原因:**
- モデルが復帰していない（持続的な推論エラー）
- 推論タイムアウトが継続している

**確認:**
```bash
# ヘルスチェック情報を監視
ros2 topic echo /model_health

# 出力例:
# data: [0.0, 0.025, 5, 50]  # health=0(不健康), time=25ms, failures=5, total=50
```

**対処:**
1. モデルの異常を特定してログから確認
2. 推論タイムアウトが解消されるまで待つ
3. 必要に応じて Jetson を再起動

```python
# config.py の FALLBACK_PP_TIMEOUT_SEC を調整
# デフォルト: 10秒 → 5秒に短縮して素早く停止
FALLBACK_PP_TIMEOUT_SEC = 5.0
```

---

### 🟡 LiDAR ノイズによる不安定な走行

**症状:**
- 障害物がないのに突然減速
- 走行ルートが左右にぐらぐらしている

**原因:**
1. LiDAR が反射の多い環境 (ガラス壁、金属) で反応
2. LiDAR 自体のノイズ・異常値

**対処:**
```python
# src/f1_env.py の LiDAR クリーニング強化
scans_clean = np.nan_to_num(
    scans_raw,
    nan=30.0,
    posinf=30.0,
    neginf=0.0
)

# ノイズフィルター追加
scans_filtered = gaussian_filter1d(scans_clean, sigma=1.0)  # 平滑化
```

**確認：**
```bash
# 生の LiDAR データを確認
ros2 topic echo /scan | head -20

# 異常な値 (inf, -inf, 0 に近い値) が多い場合は LiDAR を疑う
```

---

### 🟡 ROS2 トピック通信遅延

**症状:**
- `/safe_action` の更新が遅い (100ms 以上の遅延)
- モーターの応答に遅れが生じる

**原因:**
1. ネットワークが混雑している
2. ROS2 の QoS 設定が不適切
3. ノードのスピナーが追い付いていない

**対処:**
```python
# sharing/ros2_safety_node.py の QoS 設定を改善
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,  # UDP ライク
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

self.safe_action_pub = self.create_publisher(
    Float32MultiArray,
    '/safe_action',
    qos
)
```

**確認：**
```bash
# トピック遅延の測定
ros2 topic hz /safe_action
# 40Hz (25ms 周期) より遅い場合は要改善
```

---

## よくある質問 (FAQ)

### Q1: 実機走行でモデルが使われているのか確認したい

**A:**
```bash
# `/model_health` トピックを監視
ros2 topic echo /model_health

# 出力が変わり続ければモデルが推論中
# data: [1.0, 0.015, 0, ...]  # health=1(健康), time=15ms
```

または

```python
# ログファイルから推論の履歴を確認
import json
for line in open('logs/safety_events/safety_log_*.jsonl'):
    event = json.loads(line)
    if event['event_type'] == 'MODEL_INFERENCE':
        print(event)
```

---

### Q2: フォールバック (Pure Pursuit) のみで走行させたい

**A:**

`src/config.py` でモデルを無効化：

```python
# モデルを使用しない設定に切り替え
ENABLE_SAFETY_MANAGER = True

# 常に Pure Pursuit で走行させるため、
# 意図的にモデル推論をスキップ

# 別案: 学習済みモデルを使わず、Pure Pursuit 直結で走行
# src/f1_env.py の step() で:
action = self.pp_controller.compute_action(robot_state)
```

---

### Q3: 衝突復帰のバック走行がうまくいかない

**A:**

バック走行時の速度・時間を調整：

```python
# src/collision_recovery.py
if self.state == "BACKING":
    if phase_elapsed < 1.5:
        # 速度を上げる
        return (0.0, -0.5)  # -0.3 から -0.5 に変更
    else:
        self.state = "TURNING"
```

または、バック走行を省略して回転から開始：

```python
def start_recovery(self, x, y, yaw):
    # BACKING をスキップして TURNING から開始
    self.state = "TURNING"
    self.phase_start_time = time.time()
```

---

### Q4: 段階的降速の速度を変更したい

**A:**

`src/config.py` で `FALLBACK_SPEED_SCHEDULE` を編集：

```python
# デフォルト
FALLBACK_SPEED_SCHEDULE = [1.0, 0.8, 0.5, 0.2, 0.0]

# より早く停止 (3段階)
FALLBACK_SPEED_SCHEDULE = [1.0, 0.5, 0.0]

# より緩やかに低下 (7段階)
FALLBACK_SPEED_SCHEDULE = [1.0, 0.85, 0.7, 0.55, 0.4, 0.2, 0.0]
```

---

### Q5: 緊急停止がシステムに反映されない

**A:**

緊急停止トピック `/emergency_stop` が正しく受信されているか確認：

```bash
# Terminal 1: Safety Node を起動
python3 sharing/jetson_main.py

# Terminal 2: 緊急停止信号を送信
python3 -c "
import rclpy
from std_msgs.msg import Bool

rclpy.init()
node = rclpy.create_node('test')
pub = node.create_publisher(Bool, '/emergency_stop', 10)

msg = Bool(data=True)
pub.publish(msg)
print('Emergency stop published')

rclpy.shutdown()
"

# Terminal 1 のログで確認
# [ERROR] [ros2_safety_node]: Emergency stop activated!
```

---

### Q6: テスト時だけモデルを无視したい

**A:**

ダミーモデルを使用：

```python
# src/f1_env.py の step() 内で
if config.DUMMY_MODEL_MODE:
    # ランダムアクション
    action = np.random.uniform(-1, 1, 2)
else:
    # 実際のモデル推論
    action = model.predict(obs)[0]
```

`config.py` に追加：

```python
DUMMY_MODEL_MODE = False  # テスト時に True に変更
```

---

### Q7: ログを有効にしたい

**A:**

`src/f1_env.py` で ログ出力を有効化：

```python
def log_safety_event(self, event_type: str, details: dict = None):
    import json
    from pathlib import Path
    import time
    
    log_dir = Path('logs/safety_events')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    event = {
        'timestamp': time.time(),
        'event_type': event_type,
        'safety_state': self.safety_manager.state.name if self.safety_manager else 'N/A',
        'details': details or {}
    }
    
    log_file = log_dir / f'safety_log_{time.strftime("%Y%m%d_%H%M%S")}.jsonl'
    with open(log_file, 'a') as f:
        f.write(json.dumps(event) + '\n')

# 各イベント時に呼び出し
if model_healthy == False:
    self.log_safety_event('MODEL_FAILURE', {
        'error_info': error_info,
        'inference_time': elapsed
    })
```

---

### Q8: 複数台の F1Tenth を同時走行させたい

**A:**

ROS2 の `ROS_DOMAIN_ID` を変更して隔離：

```bash
# Robot 1
export ROS_DOMAIN_ID=1
python3 sharing/jetson_main.py

# Robot 2 (別マシン)
export ROS_DOMAIN_ID=2
python3 sharing/jetson_main.py
```

---

## 本番走行チェックリスト (最終版)

走行 **30 分以内** に以下を確認してください：

### ✅ ハードウェア
- [ ] Jetson 温度 < 70°C
- [ ] バッテリー電圧: 11.5V 以上
- [ ] LiDAR スキャン周波数: 40Hz
- [ ] モーター制御 (VESC): 反応良好

### ✅ ソフトウェア
- [ ] ONNX 推論: < 18ms
- [ ] ROS2 ノード: 全て起動
- [ ] `/action`, `/safe_action` トピック: 通信OK
- [ ] `/model_health`: health=1 (正常)

### ✅ 安全機能
- [ ] Pure Pursuit フォールバック: テスト OK
- [ ] 衝突復帰シーケンス: テスト OK (シミュレーション)
- [ ] 緊急停止: トピック発行で即停止確認
- [ ] 段階的降速: タイムアウト時に機能確認

### ✅ ロジスティクス
- [ ] コース周辺に障害物がない
- [ ] スタート地点が明確
- [ ] カメラ / スマートフォンで記録準備
- [ ] 人員配置: 誰か１人は常に見守り状態

---

## 緊急時の対応

### 走行中に異常が発生した場合

1. **緊急停止ボタンを押す** (即座に停止)
   ```bash
   python3 emergency_stop_publisher.py
   ```

2. **Jetson を再起動** (リセット)
   ```bash
   ssh jetson@<ip_address>
   sudo reboot
   ```

3. **ログを確認** (原因特定)
   ```bash
   tail -f logs/safety_events/safety_log_*.jsonl
   ```

4. **本番走行を一時停止** して原因を究明してから再開

---

## 次のステップ

→ [本番走行](00_IMPLEMENTATION_OVERVIEW.md) を参照
