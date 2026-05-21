#!/bin/bash

# EXP-40: "CNN Policy Stability Verification"
#
# 目的: developmentブランチで実装された Conv1D ポリシーと高解像度観測の動作確認。
#      「まずは安定性」をとるため、最高速度は 2.5m/s に抑え、学習の収束を確認する。
#
# 変更点:
#   1. USE_CNN_POLICY = True (src/config.py で設定)
#   2. MAX_SPEED = 2.5m/s (安定重視)
#   3. LIDAR_DOWNSAMPLE_FACTOR = 5 (高解像度: 216点)
#
# 学習手法: Fresh Start (新規学習)

MODEL_NAME="ppo_10M_exp40_cnn_verify"
LOG_FILE="/workspace/logs/exp40_output.log"

cd /workspace

echo "=== EXP-40 TRAINING STARTED (CNN Policy Verification) ===" | tee $LOG_FILE
echo "  Configuration: CNN Policy, MAX_SPEED=2.5, High Res" | tee -a $LOG_FILE
echo "  Note: Fresh Start" | tee -a $LOG_FILE

# 新規学習を実行
# 環境変数で安定志向のパラメータを注入
TRAINING_PROFILE=auto \
MAP_PATH=/workspace/my_maps/testmap-tamoku/map-tamoku \
MAX_SPEED=2.5 \
MIN_SPEED=0.3 \
python3 scripts/train.py \
    --steps 10000000 \
    --model /workspace/models/$MODEL_NAME \
    2>&1 | tee -a $LOG_FILE

echo "=== EXP-40 TRAINING FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-40 EVALUATION STARTED ===" | tee -a $LOG_FILE
MAP_PATH=/workspace/my_maps/testmap-tamoku/map-tamoku \
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-40 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-40 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
MAP_PATH=/workspace/my_maps/testmap-tamoku/map-tamoku \
python3 scripts/enjoy_wide.py \
    --model $MODEL_NAME \
    --steps 1000 \
    --save /workspace/gif/${MODEL_NAME}.mp4 \
    2>&1 | tee -a $LOG_FILE
echo "=== EXP-40 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
