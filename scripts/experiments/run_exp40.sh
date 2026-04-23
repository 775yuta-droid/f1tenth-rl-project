#!/bin/bash

# EXP-40: "High Speed Challenge (3.5m/s)"
#
# 目的: EXP-39の安定性を維持しつつ、最高速度を 2.5m/s -> 3.5m/s へ引き上げる
#
# 変更点:
#   1. MAX_SPEED = 3.5m/s
#   2. MIN_SPEED = 0.5m/s (低速停滞防止)
#   3. REWARD_SPEED_WEIGHT = 3.0 (加速インセンティブ強化)
#
# 継承: EXP-39 (ppo_10M_exp39_high_res_early_brake) から Resume

MODEL_NAME="ppo_5M_exp40_high_speed"
RESUME_MODEL="ppo_10M_exp39_high_res_early_brake"
LOG_FILE="/workspace/logs/exp40_output.log"

cd /workspace

echo "=== EXP-40 TRAINING STARTED (High Speed Challenge) ===" | tee $LOG_FILE
echo "  Configuration: MAX_SPEED=3.5, MIN_SPEED=0.5" | tee -a $LOG_FILE
echo "  Resume from: $RESUME_MODEL" | tee -a $LOG_FILE

# Resume 学習を実行
TRAINING_PROFILE=auto \
MAP_PATH=/workspace/my_maps/testmap-tamoku/map-tamoku \
python3 scripts/train.py \
    --steps 5000000 \
    --model /workspace/models/$MODEL_NAME \
    --resume /workspace/models/$RESUME_MODEL \
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
    --steps 2000 \
    --save /workspace/gif/${MODEL_NAME}.mp4 \
    2>&1 | tee -a $LOG_FILE
echo "=== EXP-40 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
