#!/bin/bash

# EXP-19 "Guided High-Res & Multi-Pose"
MODEL_NAME="ppo_10M_exp19_guided_hr"
LOG_FILE="/workspace/logs/exp19_output.log"

echo "=== EXP-19 TRAINING STARTED ===" | tee $LOG_FILE
# High-Res, SteerBonus, PoseNoise, SpeedWeight=1.1
TRAINING_PROFILE=auto python3 scripts/train.py --steps 10000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-19 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-19 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-19 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-19 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-19 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
