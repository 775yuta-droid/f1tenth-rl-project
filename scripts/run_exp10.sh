#!/bin/bash

# EXP-10 "Racing Line" training script
MODEL_NAME="ppo_5M_exp10_racingline"
LOG_FILE="/workspace/logs/exp10_output.log"

echo "=== EXP-10 TRAINING STARTED ===" | tee -a $LOG_FILE
# リセット学習。config.pyとrewards.pyの変更が反映される。
TRAINING_PROFILE=auto python3 scripts/train.py --steps 5000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-10 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-10 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-10 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-10 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-10 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
