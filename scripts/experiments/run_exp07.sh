#!/bin/bash

# EXP-07 "Deep Brain" training script
MODEL_NAME="ppo_5M_exp07_deepbrain"
LOG_FILE="/workspace/logs/exp07_output.log"

echo "=== EXP-07 TRAINING STARTED ===" | tee -a $LOG_FILE
# 新規学習(resumeなし)で実施。config.py の NET_ARCH=[128, 128] が反映される。
TRAINING_PROFILE=auto python3 scripts/train.py --steps 5000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-07 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-07 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-07 EVALUATION FINISHED ===" | tee -a $LOG_FILE
