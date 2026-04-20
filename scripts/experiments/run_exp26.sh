#!/bin/bash

# EXP-26: "Top Speed Challenge (Break the Limit)"
# EXP-25との差分: 
#  - MIN_SPEED: 0.3 -> 1.0 (徐行禁止)
#  - MAX_SPEED: 2.5 -> 4.0 (最高速度拡大)
#  - REWARD_SPEED_WEIGHT: 1.0 -> 2.0 (速度重視)
#  - REWARD_PROGRESS_WEIGHT: 2.0 -> 4.0 (距離重視)
#  - progress_scale: 0.3 -> 0.5 (中距離でも駆動を促す)
# 目的: 高速域での走行・転舵を学習し、平均速度の飛躍的向上を目指す。

MODEL_NAME="ppo_10M_exp26_top_speed"
LOG_FILE="/workspace/logs/exp26_output.log"

echo "=== EXP-26 TRAINING STARTED ===" | tee $LOG_FILE
TORCH_NUM_THREADS=1 python3 scripts/train.py --steps 10000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-26 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-26 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-26 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-26 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-26 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
