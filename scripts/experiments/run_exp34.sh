#!/bin/bash

# EXP-34: "Strict Center & Ghost Width"
# 
# 目的: 極狭コースの完全攻略
# 戦略: 
#   1. 仮想車幅拡大 (0.19 -> 0.23): AIに「車が大きい」と思わせ、早めの回避を促す。
#   2. センターペナルティ超強化 (x20.0): 中心からズレることを絶対悪とする。
#   3. ハンドル感度向上 (x1.3): ステアリングの間に合わなさを物理的に解決。
#   4. EXP-33b の最高傑作から Resume し、仕上げを行う。

BASE_MODEL="ppo_exp33b_p3_narrow_normal"
MODEL_NAME="ppo_3M_exp34_ghost_center"
LOG_FILE="/workspace/logs/exp34_output.log"

echo "=== EXP-34 TRAINING STARTED (STRICT CENTER ADAPT) ===" | tee $LOG_FILE
echo "  Base model: $BASE_MODEL (from EXP-33b)" | tee -a $LOG_FILE
echo "  Changes: CAR_WIDTH=0.23, STEER_SENS=1.3, CENTER_PENALTY=20.0" | tee -a $LOG_FILE

# 仕上げ学習 (300万ステップ)
# 速度は MAX=2.0 (実用的速度) で固定
MAP_PATH=/workspace/my_maps/testmap-0416 \
MIN_SPEED=0.3 \
MAX_SPEED=2.0 \
TORCH_NUM_THREADS=1 \
python3 scripts/train.py \
    --steps 3000000 \
    --resume /workspace/models/$BASE_MODEL \
    --model /workspace/models/$MODEL_NAME \
    2>&1 | tee -a $LOG_FILE

echo "=== EXP-34 TRAINING FINISHED ===" | tee -a $LOG_FILE

# 評価 & 動画
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 --steps 2000 2>&1 | tee -a $LOG_FILE
