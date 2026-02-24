# モデル構造の説明（現状案）

## 1. 採用しているモデルの種類

**MLP（Multi-Layer Perceptron）** をベースとした方策ネットワークを使用

- モデル種別：全結合ニューラルネットワーク（Feedforward NN）
- 時系列処理（RNN / LSTM 等）は未使用
- 画像入力なし、LiDARベースの数値入力のみ

---

## 2. 入力（State）

### LiDAR情報
- 次元数：**108次元**
-今後540にする予定

### その他の状態量
- 車両速度（linear velocity）
- ステアリング角 or 角速度（使用環境に依存）

👉 入力はすべて **1次元ベクトルとして連結**

---

## 3. ネットワーク構造（概念）
[ State (LiDAR + 車両状態) ]
↓
Fully Connected
↓
Fully Connected
↓
Fully Connected
↓
Action 出力

- 各層は ReLU 系活性化関数
- シンプルな構造だが学習は安定
- 毎回ほぼ同一の走行挙動を示す

---

## 4. 出力（Action）

- ステアリング角（continuous）
- 速度 or 加速度（continuous）

👉 **Continuous Action Space**
（PPO / SAC 等のActor-Critic系アルゴリズムと相性が良い）

---

## 5. なぜ MLP を選択しているか

### 採用理由
- 学習が安定している（Loss が暴れにくい）
- 毎回似た挙動で再現性が高い
- 実時間推論が軽く、Jetson 実装に向く

### 割り切り
- 時系列依存（過去状態の記憶）はモデル内部では扱っていない
- 現状は「今の観測だけで最善行動を決める」設計

---

## 6. 今後の改善検討ポイント（共有用）

- 残差入力（LiDAR_t − LiDAR_{t-1}）の明示的導入
- Action の時間差分（Δaction）を学習対象にする構成
- MLP + 軽量な時間要素（Frame Stack 等）の検討

※ 現時点では **「まず安定して走れること」を最優先** として MLP を採用