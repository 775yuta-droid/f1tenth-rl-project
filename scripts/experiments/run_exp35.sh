#!/bin/bash

# EXP-35: "Back to Original Frame"
# 
# 目的: 極狭コースの完全攻略のための最適化
# 戦略: 
#   1. 物理特性のリセット: CAR_WIDTH=0.19, STEER_SENS=1.0 に戻し、AIの運転感覚の狂いを解消する。
#   2. 規律の維持: center_penalty を 10.0 とし、適度な緊張感でセンターを維持させる。
#   3. EXP-33b (ppo_exp33b_p3_narrow_normal) の最も優秀だったモデルから Resume。

BASE_MODEL="ppo_exp33b_p3_narrow_normal"
MODEL_NAME="ppo_2M_exp35_back_to_original"
LOG_FILE="/workspace/logs/exp35_output.log"

echo "=== EXP-35 TRAINING STARTED (BACK TO ORIGINAL FRAME) ===" | tee $LOG_FILE
echo "  Base model: $BASE_MODEL (from EXP-33b)" | tee -a $LOG_FILE
echo "  Changes: CAR_WIDTH=0.19, STEER_SENS=1.0, CENTER_PENALTY=10.0" | tee -a $LOG_FILE

# 仕上げ学習 (200万ステップ) - ベースの動作は安定しているため短めで様子見
MAP_PATH=/workspace/my_maps/testmap-0416 \
MIN_SPEED=0.3 \
MAX_SPEED=2.0 \
TORCH_NUM_THREADS=1 \
python3 scripts/train.py \
    --steps 2000000 \
    --resume /workspace/models/$BASE_MODEL \
    --model /workspace/models/$MODEL_NAME \
    2>&1 | tee -a $LOG_FILE

echo "=== EXP-35 TRAINING FINISHED ===" | tee -a $LOG_FILE

# 評価 & 動画
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 --steps 2000 2>&1 | tee -a $LOG_FILE
