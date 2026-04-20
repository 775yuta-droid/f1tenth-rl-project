#!/bin/bash

# EXP-15 "Prudent Speed"
MODEL_NAME="ppo_10M_exp15_prudent_speed"
LOG_FILE="/workspace/logs/exp15_output.log"

echo "=== EXP-15 TRAINING STARTED ===" | tee $LOG_FILE
# TOTAL_TIMESTEPS=10M, MIN_SPEED=0.3, REWARD_SPEED_WEIGHT=1.2, REWARD_PROGRESS_WEIGHT=2.0, LIDAR_RESIDUAL=False
TRAINING_PROFILE=auto python3 scripts/train.py --steps 10000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-15 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-15 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-15 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-15 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-15 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
