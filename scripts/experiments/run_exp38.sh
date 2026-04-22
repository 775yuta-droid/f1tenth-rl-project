#!/bin/bash

# EXP-38: "Diagonal Front Detection for Curve Entry"
#
# 目的: カーブ入口での外膨らみを解消する
# 問題: EXP-37モデルは直線は問題なく走れるが、カーブ前方が開いている間に
#       曲がり始めず、外側に蛇行する。
#
# 変更点 (rewards.py):
#   1. 前方検出角を ±60° → ±40° に絞り、中央前方の精度を向上
#   2. 斜め前方 (±40°〜±80°) の左右非対称シグナルを追加
#      - 前方が開いているのに斜め前で壁の差がある = カーブ入口
#      - カーブ方向に操舵している → ボーナス (+1.0 * alignment)
#      - 直進 or 逆操舵 → ペナルティ (-1.5 * asymmetry)
#   3. ステアリング安定化ペナルティを 0.3 → 0.1 に軽減
#      (カーブでの積極的操舵を妨げない)
#
# 継続学習: EXP-37の観測次元 (81+2=83dim) は変わらないためResumeが可能

BASE_MODEL="ppo_10M_exp37_tamoku_270deg_fresh"
MODEL_NAME="ppo_10M_exp38_curve_entry"
LOG_FILE="/workspace/logs/exp38_output.log"

cd /workspace

echo "=== EXP-38 TRAINING STARTED (Diagonal Front / Curve Entry Detection) ===" | tee $LOG_FILE
echo "  Base model : $BASE_MODEL (from EXP-37)" | tee -a $LOG_FILE
echo "  Key change : Diagonal front asymmetry reward + Reduced steer penalty" | tee -a $LOG_FILE

# EXP-37のモデルからResumeして10Mステップ追加学習
TRAINING_PROFILE=auto \
MAP_PATH=/workspace/my_maps/testmap-tamoku/map-tamoku \
python3 scripts/train.py \
    --steps 10000000 \
    --resume /workspace/models/$BASE_MODEL \
    --model /workspace/models/$MODEL_NAME \
    2>&1 | tee -a $LOG_FILE

echo "=== EXP-38 TRAINING FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-38 EVALUATION STARTED ===" | tee -a $LOG_FILE
MAP_PATH=/workspace/my_maps/testmap-tamoku/map-tamoku \
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-38 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-38 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
MAP_PATH=/workspace/my_maps/testmap-tamoku/map-tamoku \
python3 scripts/enjoy_wide.py \
    --model $MODEL_NAME \
    --steps 2000 \
    --save /workspace/gif/${MODEL_NAME}.mp4 \
    2>&1 | tee -a $LOG_FILE
echo "=== EXP-38 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
