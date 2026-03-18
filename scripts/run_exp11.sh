#!/bin/bash

# EXP-11 "Fast & Boundary" training script
MODEL_NAME="ppo_10M_exp11_boundary"
LOG_FILE="/workspace/logs/exp11_output.log"

echo "=== EXP-11 TRAINING STARTED ===" | tee -a $LOG_FILE
# TOTAL_TIMESTEPS=10M, LEARNING_RATE=5e-5, ExpWallPenalty
TRAINING_PROFILE=auto python3 scripts/train.py --steps 10000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-11 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-11 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-11 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-11 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-11 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
