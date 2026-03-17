#!/bin/bash

# EXP-08 "Safe Cornering" training script
MODEL_NAME="ppo_5M_exp08_safecorner"
LOG_FILE="/workspace/logs/exp08_output.log"

echo "=== EXP-08 TRAINING STARTED ===" | tee -a $LOG_FILE
# 新規学習(resumeなし)。config.py の SAFETY_WEIGHT=1.5 が反映される。
TRAINING_PROFILE=auto python3 scripts/train.py --steps 5000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-08 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-08 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-08 EVALUATION FINISHED ===" | tee -a $LOG_FILE
