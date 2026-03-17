#!/bin/bash

# EXP-09 "Inner Wall Fix" training script
MODEL_NAME="ppo_5M_exp09_innerwall"
LOG_FILE="/workspace/logs/exp09_output.log"

echo "=== EXP-09 TRAINING STARTED ===" | tee -a $LOG_FILE
# INCLUDE_LIDAR_RESIDUAL=False, SAFETY_WEIGHT=1.0, NET_ARCH=[128,128]
# rewards.py に左右非対称ペナルティを追加済み
TRAINING_PROFILE=auto python3 scripts/train.py --steps 5000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-09 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-09 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-09 EVALUATION FINISHED ===" | tee -a $LOG_FILE
