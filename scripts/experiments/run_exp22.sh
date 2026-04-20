#!/bin/bash

# EXP-22 "Back to Basics + Parallel Training"
# - STEER_SENSITIVITY=1.0 (EXP-13/16の成功設定に復帰)
# - SubprocVecEnv 8並列 (学習高速化: 3.5h -> ~40min)
# - シンプルな報酬設計 (EXP-16ベース)
# - フレーム積層 4frames, mean() サンプリング維持

MODEL_NAME="ppo_10M_exp22_parallel"
LOG_FILE="/workspace/logs/exp22_output.log"

echo "=== EXP-22 TRAINING STARTED ===" | tee $LOG_FILE
# TORCH_NUM_THREADS=1 を強制 (8並列プロセス × 1スレッド = CPU効率最大化)
TORCH_NUM_THREADS=1 python3 scripts/train.py --steps 10000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-22 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-22 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-22 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-22 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-22 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
