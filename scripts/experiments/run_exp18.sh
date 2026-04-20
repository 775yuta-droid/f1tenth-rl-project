#!/bin/bash

# EXP-18 "Resolution & Steering Sensitivity Optimization"
MODEL_NAME="ppo_10M_exp18_high_res"
LOG_FILE="/workspace/logs/exp17_output.log" # ログファイル名は17のまま流用（上書き注意） or exp18に
LOG_FILE="/workspace/logs/exp18_output.log"

echo "=== EXP-18 TRAINING STARTED ===" | tee $LOG_FILE
# LIDAR_DOWNSAMPLE_FACTOR=4, STEER_SENSITIVITY=0.41, Rewards=EXP-16 style
TRAINING_PROFILE=auto python3 scripts/train.py --steps 10000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-18 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-18 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-18 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-18 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-18 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
