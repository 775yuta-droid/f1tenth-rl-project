#!/bin/bash

# EXP-13 "Brake Before Corner"
MODEL_NAME="ppo_10M_exp13_brake"
LOG_FILE="/workspace/logs/exp13_output.log"

echo "=== EXP-13 TRAINING STARTED ===" | tee -a $LOG_FILE
# TOTAL_TIMESTEPS=10M, MIN_SPEED=0.3, SharpCornerBrakingPenalty
TRAINING_PROFILE=auto python3 scripts/train.py --steps 10000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-13 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-13 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-13 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-13 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-13 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
