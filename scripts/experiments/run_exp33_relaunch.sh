#!/bin/bash

# EXP-33b: "Relaunch Curriculum - Recovery from R1"
#
# 昨晩の学習プロセスが途絶えたため、改善を加えて再起動。
# 修正点:
#   1. REWARD_COLLISION = -50 (勾配を滑らかにし、初期の衝突で学習が止まらないように)
#   2. nohup によるプロセス保護
#   3. Phase 1 をスキップし、EXP-25 から直接 testmap-0416 (低速) へ。

LOG_FILE="/workspace/logs/exp33b_output.log"
BASE_MODEL="ppo_10M_exp25_fast_stable"
P2_MODEL="ppo_exp33b_p2_narrow_slow"
P3_MODEL="ppo_exp33b_p3_narrow_normal"

echo "=== EXP-33b RELAUNCH STARTED ===" | tee $LOG_FILE
echo "  REWARD_COLLISION: -50.0" | tee -a $LOG_FILE

# Phase 2: testmap-0416 / Low Speed
echo "[Phase 2] testmap-0416 (MAX=1.5) Starting..." | tee -a $LOG_FILE
MAP_PATH=/workspace/my_maps/testmap-0416 \
MIN_SPEED=0.3 \
MAX_SPEED=1.5 \
TORCH_NUM_THREADS=1 \
python3 scripts/train.py \
    --steps 3000000 \
    --resume /workspace/models/$BASE_MODEL \
    --model /workspace/models/$P2_MODEL \
    2>&1 | tee -a $LOG_FILE

# Phase 3: testmap-0416 / Normal Speed
echo "[Phase 3] testmap-0416 (MAX=2.0) Starting..." | tee -a $LOG_FILE
MAP_PATH=/workspace/my_maps/testmap-0416 \
MIN_SPEED=0.3 \
MAX_SPEED=2.0 \
TORCH_NUM_THREADS=1 \
python3 scripts/train.py \
    --steps 2000000 \
    --resume /workspace/models/$P2_MODEL \
    --model /workspace/models/$P3_MODEL \
    2>&1 | tee -a $LOG_FILE

echo "=== EXP-33b ALL FINISHED ===" | tee -a $LOG_FILE

# 最終評価
python3 scripts/evaluate.py --model $P3_MODEL --episodes 20 2>&1 | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $P3_MODEL --save /workspace/gif/${P3_MODEL}.mp4 --steps 1500 2>&1 | tee -a $LOG_FILE
