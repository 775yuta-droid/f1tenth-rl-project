#!/bin/bash

# EXP-29: "Straight Fix (Remove Steer Penalty + Partial Brake)"
# EXP-28との差分（config.py / rewards.py を参照）:
#  - rewards.py: ステア連動速度ペナルティを完全削除
#      【EXP-28の失敗分析】 steer_intensityに応じた最大8倍のペナルティが
#      「大舵角 = 不利」という誤学習を招き、直進→壁衝突を引き起こした。
#      シンプルな一定倍率(2.0)のペナルティに戻す。
#  - config.py: MIN_SPEED: 1.0 -> 0.5
#      EXP-26以降、MIN_SPEED=1.0によりコーナーで物理的に曲がれなかった。
#      0.5まで緩和してコーナリング能力を回復。0.3に戻さず局所解を防ぐ。
#  - LiDAR解像度 (DOWNSAMPLE=5, 216点) と FRAME_STACK=2 はEXP-28設定を維持
# 目的: EXP-28で発生した「直進して壁衝突」の根本原因を修正し、
#       高解像度LiDARを維持しながらコーナリング能力を回復させる。

MODEL_NAME="ppo_10M_exp29_straight_fix"
LOG_FILE="/workspace/logs/exp29_output.log"

echo "=== EXP-29 TRAINING STARTED (FRESH) ===" | tee $LOG_FILE
echo "  Changes: Remove steer penalty + MIN_SPEED=0.5" | tee -a $LOG_FILE
echo "  Model: $MODEL_NAME" | tee -a $LOG_FILE
# 観測次元はEXP-28と同じ(DOWNSAMPLE=5)のため、理論上Resumeも可能だが
# 報酬設計の変更が大きいため新規学習を推奨
TORCH_NUM_THREADS=1 python3 scripts/train.py --steps 10000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-29 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-29 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-29 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-29 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-29 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
