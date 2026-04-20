#!/bin/bash

# EXP-36: "Snail Paced Survival"
# 
# 目的: 極狭コースでの「初の完走(100%)」達成
# 戦略: 
#   1. MIN_SPEED=0.1, MAX_SPEED=0.5 に極制限し、AIが壁を避ける「時間的余裕」を无限に与える。
#   2. EXP-35 のモデル (ppo_2M_exp35_back_to_original) から Resume する。
#   3. とにかく「生き残れば完走できる」ことを脳に刻み付けさせる。

BASE_MODEL="ppo_2M_exp35_back_to_original"
MODEL_NAME="ppo_2M_exp36_snail_pace"
LOG_FILE="/workspace/logs/exp36_output.log"

echo "=== EXP-36 TRAINING STARTED (SNAIL PACED SURVIVAL) ===" | tee $LOG_FILE
echo "  Base model: $BASE_MODEL (from EXP-35)" | tee -a $LOG_FILE
echo "  Changes: MIN_SPEED=0.1, MAX_SPEED=0.5" | tee -a $LOG_FILE

# 確実完走フェーズ (200万ステップ)
MAP_PATH=/workspace/my_maps/testmap-0416 \
MIN_SPEED=0.1 \
MAX_SPEED=0.5 \
TORCH_NUM_THREADS=1 \
python3 scripts/train.py \
    --steps 2000000 \
    --resume /workspace/models/$BASE_MODEL \
    --model /workspace/models/$MODEL_NAME \
    2>&1 | tee -a $LOG_FILE

echo "=== EXP-36 TRAINING FINISHED ===" | tee -a $LOG_FILE

# 評価 & 動画 (超低速のため動画ステップ数を増やして対応)
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 --steps 5000 2>&1 | tee -a $LOG_FILE
