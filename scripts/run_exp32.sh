#!/bin/bash

# EXP-32: "Narrow Adapt (Resume from EXP-25)"
# 目的: 実機マップ(testmap-0416)の「極狭直線・極狭カーブ」を攻略する。
#
# EXP-31の失敗からの学び:
#   - Fresh学習では5Mステップでも完走率0%（平均48ステップで衝突）
#   - 道幅がマシンの1.8倍という超高難易度コースに「初心者」から挑むのは無謀
#   - 対策: EXP-25（完走率100%）の「基本操舵」を転用し、狭路特化で微調整する
#
# EXP-32での変更点（EXPERIMENT_PLAN.md 参照）:
#   rewards.py:
#     - Hokuyo 270° マスキング: scans[135:945] で後方90°を除外 (Sim-to-Real)
#     - センターライン: center_bonus(+0.3) → center_penalty(二乗×3.0) に変更
#     - 側面壁デッドライン: 0.6m → 0.35m (マシン幅0.19m+安全マージン)
#     - 狭いカーブ複合ペナルティ: front<4.5m & side<0.7m で speed×4.0罰
#     - ステア安定性係数: 0.3 → 0.4
#   config.py:
#     - MIN_SPEED: 0.5 → 0.3 (EXP-13ブレイクスルー設定。コーナーで減速できる余地)
#     - MAX_SPEED: 3.0 → 2.0 (完走最優先。速さより精度)
#     - LEARNING_RATE: 5e-5 → 2e-5 (EXP-25の知識を壊さない)
#     - REWARD_SURVIVAL: 0.2 → 0.5 (長生きすること自体を最大インセンティブに)
#     - PPO_ENT_COEF: 0.03 → 0.01 (Resume時の探索を絞る)

BASE_MODEL="ppo_10M_exp25_fast_stable"
MODEL_NAME="ppo_3M_exp32_narrow_adapt"
LOG_FILE="/workspace/logs/exp32_output.log"

echo "=== EXP-32 TRAINING STARTED (NARROW ADAPT: Resume from EXP-25) ===" | tee $LOG_FILE
echo "  Base model: $BASE_MODEL" | tee -a $LOG_FILE
echo "  Map: testmap-0416 (道幅1.8倍の極狭コース)" | tee -a $LOG_FILE
echo "  Changes: Hokuyo270°, MIN=0.3, MAX=2.0, SURVIVAL=0.5, LR=2e-5" | tee -a $LOG_FILE
echo "  Strategy: センターライン二乗ペナルティ + 狭カーブ複合ペナルティ" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# EXP-25の観測空間(DOWNSAMPLE=10, FRAME_STACK=4)と完全一致でResume
TORCH_NUM_THREADS=1 python3 scripts/train.py \
    --steps 3000000 \
    --resume /workspace/models/$BASE_MODEL \
    --model /workspace/models/$MODEL_NAME \
    2>&1 | tee -a $LOG_FILE

echo "=== EXP-32 TRAINING FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-32 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-32 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-32 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py \
    --model $MODEL_NAME \
    --save /workspace/gif/${MODEL_NAME}.mp4 \
    --steps 2000 \
    2>&1 | tee -a $LOG_FILE
echo "=== EXP-32 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
