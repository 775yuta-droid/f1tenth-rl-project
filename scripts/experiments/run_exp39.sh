#!/bin/bash

# EXP-39: "High Resolution & Early Brake"
#
# 目的: カーブで外側に膨らんで衝突する問題を解決する
#
# 変更点:
#   1. LIDAR_DOWNSAMPLE_FACTOR = 5 (解像度2倍: 1.67°/点)
#   2. ブレーキ開始距離を 2.0m -> 4.0m に拡大
#   3. EXP-38 で導入した「斜め前方非対称報酬」を継続使用
#
# 注意: 次元変更 (332 -> 656) のため、Fresh Start で学習します

MODEL_NAME="ppo_10M_exp39_high_res_early_brake"
LOG_FILE="/workspace/logs/exp39_output.log"

cd /workspace

echo "=== EXP-39 TRAINING STARTED (High Res / Early Brake) ===" | tee $LOG_FILE
echo "  Configuration: Factor 5, Brake TH 4.0m" | tee -a $LOG_FILE
echo "  Note: Fresh Start (New dimensions)" | tee -a $LOG_FILE

# 解像度変更のため必ず新規学習
TRAINING_PROFILE=auto \
MAP_PATH=/workspace/my_maps/testmap-tamoku/map-tamoku \
python3 scripts/train.py \
    --steps 10000000 \
    --model /workspace/models/$MODEL_NAME \
    2>&1 | tee -a $LOG_FILE

echo "=== EXP-39 TRAINING FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-39 EVALUATION STARTED ===" | tee -a $LOG_FILE
MAP_PATH=/workspace/my_maps/testmap-tamoku/map-tamoku \
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-39 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-39 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
MAP_PATH=/workspace/my_maps/testmap-tamoku/map-tamoku \
python3 scripts/enjoy_wide.py \
    --model $MODEL_NAME \
    --steps 2000 \
    --save /workspace/gif/${MODEL_NAME}.mp4 \
    2>&1 | tee -a $LOG_FILE
echo "=== EXP-39 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
