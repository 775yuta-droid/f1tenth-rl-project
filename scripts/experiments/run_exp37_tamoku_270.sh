#!/bin/bash

# EXP-37: "Fresh Start with 270deg Observation on Tamoku Map"
# - observation: 270° (810 points / Factor 10 = 81dim + 2 state = 83dim)
# - reward: Reverted to EXP-25 Baseline (Simple Speed + Centerline Penalty)
# - map: testmap-tamoku (Large space / Straight dominant)
# - physics: MAX_SPEED=2.5, MIN_SPEED=0.3

MODEL_NAME="ppo_10M_exp37_tamoku_270deg_fresh"
LOG_FILE="/workspace/logs/exp37_output.log"

# プロジェクトルート（/workspace）に移動して実行
cd /workspace

echo "=== EXP-37 TRAINING STARTED ===" | tee -a $LOG_FILE
# 観測次元数が変更されたため、必ず新規学習(Fresh)で行う必要があります
TRAINING_PROFILE=auto python3 scripts/train.py --steps 20000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-37 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-37 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-37 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-37 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --steps 2000 --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-37 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
