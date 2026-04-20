#!/bin/bash

# EXP-17 "Stable Speed Up"
MODEL_NAME="ppo_10M_exp17_steer_speed"
LOG_FILE="/workspace/logs/exp17_output.log"

echo "=== EXP-17 TRAINING STARTED ===" | tee $LOG_FILE
# TOTAL_TIMESTEPS=10M, MIN_SPEED=0.4, REWARD_SPEED_WEIGHT=1.2, SteerBonus=0.2
TRAINING_PROFILE=auto python3 scripts/train.py --steps 10000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-17 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-17 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-17 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-17 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-17 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
