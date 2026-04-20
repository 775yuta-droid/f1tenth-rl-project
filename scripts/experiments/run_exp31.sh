#!/bin/bash

# EXP-31: "First Flight on testmap-0416 (Fresh Training)"
# 目的: Cartographerで作成した実機マップでの初めての学習。
# 戦略: 
#  - マップが小さいため、まずは Fresh（新規）で学習させ、このマップ独自の形状を覚えさせる。
#  - 地形に特化させるため、ステップ数は 5M とし、短時間で評価まで回す。
#  - config.py で設定した新しい START_POSE を使用。

MODEL_NAME="ppo_5M_exp31_testmap_fresh"
LOG_FILE="/workspace/logs/exp31_output.log"

echo "=== EXP-31 TRAINING STARTED (NEW MAP: testmap-0416) ===" | tee $LOG_FILE
echo "  Model name: $MODEL_NAME" | tee -a $LOG_FILE
echo "  Map path: /workspace/my_maps/testmap-0416" | tee -a $LOG_FILE

# 学習実行
# config.py の設定をベースに、500万ステップ実行
TORCH_NUM_THREADS=1 python3 scripts/train.py \
    --steps 5000000 \
    --model /workspace/models/$MODEL_NAME \
    2>&1 | tee -a $LOG_FILE

echo "=== EXP-31 TRAINING FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-31 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-31 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-31 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
# 走行動画の生成
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-31 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
