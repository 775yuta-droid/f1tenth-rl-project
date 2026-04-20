#!/bin/bash

# EXP-23: "min() sampling + Frame Stacking"
# EXP-22との差分: LiDARサンプリングを mean -> min に戻すだけ
# 目的: コーナーの壁への接近信号を正確に伝えてEXP-16(50%)超えを狙う
MODEL_NAME="ppo_10M_exp23_min_stack"
LOG_FILE="/workspace/logs/exp23_output.log"

echo "=== EXP-23 TRAINING STARTED ===" | tee $LOG_FILE
TORCH_NUM_THREADS=1 python3 scripts/train.py --steps 10000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-23 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-23 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-23 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-23 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-23 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
