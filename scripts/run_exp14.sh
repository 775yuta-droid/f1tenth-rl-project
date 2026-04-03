#!/bin/bash

# EXP-14 "Gradual Speed Up"
MODEL_NAME="ppo_10M_exp14_gradual_speedup"
LOG_FILE="/workspace/logs/exp14_output.log"

echo "=== EXP-14 TRAINING STARTED ===" | tee -a $LOG_FILE
# TOTAL_TIMESTEPS=10M, MIN_SPEED=0.3, REWARD_SPEED_WEIGHT=2.0, REWARD_PROGRESS_WEIGHT=3.0, LIDAR_RESIDUAL=True
TRAINING_PROFILE=auto python3 scripts/train.py --steps 10000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-14 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-14 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-14 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-14 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-14 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
