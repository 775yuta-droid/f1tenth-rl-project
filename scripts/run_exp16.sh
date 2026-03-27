#!/bin/bash

# EXP-16 "EXP-13 Complete Reproduction"
# 目的: EXP-13 (70%完走) の設定を完全に再現し、再現性があるかを検証する
# 設定: SPEED_WEIGHT=1.0, PROGRESS_WEIGHT=2.0, LIDAR_RESIDUAL=False (EXP-13と完全同一)
MODEL_NAME="ppo_10M_exp16_reproduce_exp13"
LOG_FILE="/workspace/logs/exp16_output.log"

echo "=== EXP-16 TRAINING STARTED ===" | tee $LOG_FILE
TRAINING_PROFILE=auto python3 scripts/train.py --steps 10000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-16 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-16 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-16 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-16 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-16 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
