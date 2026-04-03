#!/bin/bash

# EXP-20 "Steering Delta Control Optimization"
MODEL_NAME="ppo_10M_exp20_steer_delta"
LOG_FILE="/workspace/logs/exp20_output.log"

echo "=== EXP-20 TRAINING STARTED ===" | tee $LOG_FILE
# High-Res (Factor 4), SteerDelta (0.05), Rewards=DeltaPenalty
TRAINING_PROFILE=auto python3 scripts/train.py --steps 10000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-20 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-20 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-20 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-20 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-20 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
