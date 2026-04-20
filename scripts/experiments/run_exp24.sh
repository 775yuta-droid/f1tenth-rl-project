#!/bin/bash

# EXP-24: "Spawn Position Filtering"
# EXP-23との差分: 苦手位置 [0.7, 5.0, -1.0] を除外
# 目的: 初期姿勢が難しい地点での早期衝突を排除し、完走率70%超えを狙う
MODEL_NAME="ppo_10M_exp24_spawn_filtered"
LOG_FILE="/workspace/logs/exp24_output.log"

echo "=== EXP-24 TRAINING STARTED ===" | tee $LOG_FILE
TORCH_NUM_THREADS=1 python3 scripts/train.py --steps 10000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-24 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-24 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-24 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-24 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-24 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
