#!/bin/bash

# EXP-25: "Fast & Stable (Spawn Filtering + Reward Normalization)"
# EXP-24との差分: 
#  - [4.5, 4.4, 2.0] スポーン地点を追加除外
#  - ブレーキ開始距離を 3.5m -> 2.0m に緩和
#  - 減速ペナルティを 3.0 -> 2.0 に緩和
#  - 生存報酬を 0.1 -> 0.2 に増加
# 目的: 高速化と完走率100%の同時達成、および累積報酬のプラス転換。

MODEL_NAME="ppo_10M_exp25_fast_stable"
LOG_FILE="/workspace/logs/exp25_output.log"

echo "=== EXP-25 TRAINING STARTED ===" | tee $LOG_FILE
TORCH_NUM_THREADS=1 python3 scripts/train.py --steps 10000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-25 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-25 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-25 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-25 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-25 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
