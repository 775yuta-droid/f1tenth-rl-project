#!/bin/bash

MODEL_NAME="ppo_4M_exp05_hybrid"
LOG_FILE="/workspace/logs/exp05_output.log"

echo "=== EXP-05 TRAINING STARTED ===" | tee -a $LOG_FILE
python3 scripts/train.py --steps 4000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-05 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-05 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-05 EVALUATION FINISHED ===" | tee -a $LOG_FILE
