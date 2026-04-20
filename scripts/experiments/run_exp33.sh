#!/bin/bash

# EXP-33: "Curriculum Learning - Narrow Map 3-Phase Training"
#
# ============================================================
# 背景: EXP-31/32での失敗分析
#   - testmap-0416 は道幅1.8倍の極狭コース（直線もカーブも狭い）
#   - いきなり挑戦させても平均116ステップで衝突→完走率0%
#   - 「広い道のプロが狭い道で走る」段階的訓練が必要
#
# カリキュラム設計:
#   Phase 1: my_map（広い）   で新報酬に慣れさせる      [2M steps]
#   Phase 2: testmap-0416（狭い）を極低速で覚える        [3M steps]
#   Phase 3: testmap-0416（狭い）で速度を段階的に上げる  [2M steps]
# ============================================================

LOG_FILE="/workspace/logs/exp33_output.log"
BASE_MODEL="ppo_10M_exp25_fast_stable"
P1_MODEL="ppo_exp33_phase1_mymap"
P2_MODEL="ppo_exp33_phase2_narrow_slow"
P3_MODEL="ppo_exp33_phase3_narrow_normal"

echo "======================================================" | tee $LOG_FILE
echo " EXP-33: Curriculum Learning - 3 Phase Training" | tee -a $LOG_FILE
echo "======================================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# ----------------------------------------------------------
# Phase 1: my_map (広いマップ) で EXP-32 の新報酬に適応
# ----------------------------------------------------------
# 目的: EXP-25の「曲がる知識」を失わずに、
#       270°マスク済み報酬 + センターライン二乗ペナルティに慣れさせる
# 広い道なので衝突リスクが低く、新しい報酬体系を安全に学習できる
echo "[Phase 1] my_map でウォームアップ (2M steps)" | tee -a $LOG_FILE
echo "  Map: my_map  MIN: 0.3  MAX: 2.0" | tee -a $LOG_FILE

MAP_PATH=/workspace/my_maps/my_map \
MIN_SPEED=0.3 \
MAX_SPEED=2.0 \
TORCH_NUM_THREADS=1 \
python3 scripts/train.py \
    --steps 2000000 \
    --resume /workspace/models/$BASE_MODEL \
    --model /workspace/models/$P1_MODEL \
    2>&1 | tee -a $LOG_FILE

echo "[Phase 1] 完了" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# ----------------------------------------------------------
# Phase 2: testmap-0416 (狭いマップ) を極低速で初体験
# ----------------------------------------------------------
# 目的: 「広いマップで新報酬を習得したモデル」を使い、
#       狭いマップを MAX=1.5m/s の超低速で走らせ、「まず完走」を目指す
# 速度が低いため、コーナーでも余裕を持って曲がりやすい
echo "[Phase 2] testmap-0416 を低速で学習 (3M steps)" | tee -a $LOG_FILE
echo "  Map: testmap-0416  MIN: 0.3  MAX: 1.5" | tee -a $LOG_FILE

MAP_PATH=/workspace/my_maps/testmap-0416 \
MIN_SPEED=0.3 \
MAX_SPEED=1.5 \
TORCH_NUM_THREADS=1 \
python3 scripts/train.py \
    --steps 3000000 \
    --resume /workspace/models/$P1_MODEL \
    --model /workspace/models/$P2_MODEL \
    2>&1 | tee -a $LOG_FILE

echo "[Phase 2] 完了 - 評価を実施" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $P2_MODEL --episodes 20 2>&1 | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# ----------------------------------------------------------
# Phase 3: testmap-0416 で通常速度に引き上げ
# ----------------------------------------------------------
# 目的: Phase 2 で「低速完走」ができるようになったモデルを
#       MAX=2.0m/s に引き上げ、実用速度での走行を学習させる
echo "[Phase 3] testmap-0416 を通常速度で学習 (2M steps)" | tee -a $LOG_FILE
echo "  Map: testmap-0416  MIN: 0.3  MAX: 2.0" | tee -a $LOG_FILE

MAP_PATH=/workspace/my_maps/testmap-0416 \
MIN_SPEED=0.3 \
MAX_SPEED=2.0 \
TORCH_NUM_THREADS=1 \
python3 scripts/train.py \
    --steps 2000000 \
    --resume /workspace/models/$P2_MODEL \
    --model /workspace/models/$P3_MODEL \
    2>&1 | tee -a $LOG_FILE

echo "[Phase 3] 完了" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# ----------------------------------------------------------
# 最終評価 & 動画生成
# ----------------------------------------------------------
echo "=== EXP-33 最終評価 (Phase 3 モデル) ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $P3_MODEL --episodes 20 2>&1 | tee -a $LOG_FILE

echo "=== EXP-33 動画生成 ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py \
    --model $P3_MODEL \
    --save /workspace/gif/${P3_MODEL}.mp4 \
    --steps 2000 \
    2>&1 | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "======================================================" | tee -a $LOG_FILE
echo " EXP-33: すべてのフェーズが完了しました" | tee -a $LOG_FILE
echo "======================================================" | tee -a $LOG_FILE
