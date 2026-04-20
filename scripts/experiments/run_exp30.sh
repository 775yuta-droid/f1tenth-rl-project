#!/bin/bash

# EXP-30: "Gradual Speed Up (Clean Resume from EXP-25)"
# 戦略転換: EXP-26〜29の連続失敗の教訓から、新規学習を捨てResumeに切り替える。
#
# EXP-25との差分（config.py を参照）:
#  - 学習方法: Fresh → Resume (ppo_10M_exp25_fast_stable から継続)
#    * EXP-25は100%完走達成済み。「曲がる能力」を保持したまま高速化のみ学習させる。
#  - LIDAR_DOWNSAMPLE_FACTOR: 10 (EXP-25と完全一致。観測次元の不整合を防ぐ)
#  - FRAME_STACK: 4 (EXP-25と完全一致。108点×4 = 434次元で整合)
#  - MIN_SPEED: 0.3 -> 0.5 (完全停車による局所解を防ぎつつ、最低限の速度を保持)
#  - MAX_SPEED: 2.5 -> 3.0 (段階的な引き上げ。EXP-26の4.0は大きすぎた)
#  - rewards.py: ステアペナルティなし (EXP-28の失敗要因を引き継がない)
#
# 失敗の系譜:
#  EXP-26: MIN=1.0, MAX=4.0, Fresh → 0% (ブレーキ不能で曲がれない)
#  EXP-27: EXP-26 + 探索強化, Resume → 0% (同上)
#  EXP-28: EXP-27 + ステアペナルティ → 0% (直進して壁衝突)
#  EXP-29: EXP-28 - ステアペナルティ, MIN=0.5, Fresh → 0% (std=20.4 発散)
#  EXP-30: EXP-25から Resume, MIN=0.5, MAX=3.0 → 期待: 曲がる能力を保持しつつ高速化

BASE_MODEL="ppo_10M_exp25_fast_stable"
MODEL_NAME="ppo_5M_exp30_resume_speed"
LOG_FILE="/workspace/logs/exp30_output.log"

echo "=== EXP-30 TRAINING STARTED (RESUME FROM EXP-25) ===" | tee $LOG_FILE
echo "  Base model: $BASE_MODEL" | tee -a $LOG_FILE
echo "  Changes: MIN_SPEED=0.5, MAX_SPEED=3.0, Resume" | tee -a $LOG_FILE
# EXP-25の観測空間(DOWNSAMPLE=10, FRAME_STACK=4)と完全一致させてResumeする
# 5Mステップ: Resumeなので新規学習(10M)より短く設定
TORCH_NUM_THREADS=1 python3 scripts/train.py \
    --steps 5000000 \
    --resume /workspace/models/$BASE_MODEL \
    --model /workspace/models/$MODEL_NAME \
    2>&1 | tee -a $LOG_FILE

echo "=== EXP-30 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-30 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-30 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-30 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-30 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
