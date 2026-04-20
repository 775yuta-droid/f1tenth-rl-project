#!/bin/bash

# EXP-27: "Exploration-heavy Resume"
# EXP-26との差分: 
#  - Resume from EXP-26 model (ppo_10M_exp26_top_speed)
#  - PPO_ENT_COEF: 0.01 -> 0.05 (探索を大幅に強化)
# 目的: 高速域（1.0m/s以上）での転舵タイミングを修正し、完走率を回復させる。

MODEL_NAME="ppo_20M_exp27_resume_explore"
RESUME_MODEL="/workspace/models/ppo_10M_exp26_top_speed"
LOG_FILE="/workspace/logs/exp27_output.log"

echo "=== EXP-27 TRAINING STARTED (RESUME) ===" | tee $LOG_FILE
TORCH_NUM_THREADS=1 python3 scripts/train.py \
    --resume $RESUME_MODEL \
    --steps 10000000 \
    --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-27 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-27 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-27 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-27 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-27 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
