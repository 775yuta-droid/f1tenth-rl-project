#!/bin/bash

# EXP-12 "True Dimension Training"
MODEL_NAME="ppo_10M_exp12_truedim"
LOG_FILE="/workspace/logs/exp12_output.log"

echo "=== EXP-12 TRAINING STARTED ===" | tee -a $LOG_FILE
# TOTAL_TIMESTEPS=10M, LEARNING_RATE=5e-5 (from config), ExpWallPenalty (from rewards)
# Car dimensions (0.465x0.19) are now forced in f1_env.py
TRAINING_PROFILE=auto python3 scripts/train.py --steps 10000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-12 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-12 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-12 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-12 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-12 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
