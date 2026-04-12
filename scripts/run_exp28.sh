#!/bin/bash

# EXP-28: "Precision Turn (High-Res & Low-Latency)"
# EXP-27との差分: 
#  - LIDAR_DOWNSAMPLE_FACTOR: 10 -> 5 (216点に倍増)
#  - FRAME_STACK: 4 -> 2 (判断遅延の削減)
#  - Steering-dependent Speed Penalty: 大舵角時の速度抑制を強化
# 目的: マシンが「曲がる意思」を適切に「実行」できるように、視力と反応速度を改善する。

MODEL_NAME="ppo_10M_exp28_precision_turn"
LOG_FILE="/workspace/logs/exp28_output.log"

echo "=== EXP-28 TRAINING STARTED (FRESH) ===" | tee $LOG_FILE
# 観測次元が変わるため、Resumeはせず新規学習を開始
TORCH_NUM_THREADS=1 python3 scripts/train.py --steps 10000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-28 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-28 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-28 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-28 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-28 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
