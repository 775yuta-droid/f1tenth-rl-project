#!/bin/bash
set -e

echo "======================================"
echo " Starting EXP-14 (Speed Up & Brake) "
echo "======================================"

echo "1. Training (4,000,000 steps)..."
TRAINING_PROFILE=auto python3 scripts/train.py --steps 4000000 --model /workspace/models/ppo_exp14

echo "2. Evaluating (20 episodes)..."
python3 scripts/evaluate.py --model ppo_exp14 --episodes 20

echo "3. Generating Video (enjoy_wide)..."
python3 scripts/enjoy_wide.py --model ppo_exp14 --save /workspace/gif/ppo_exp14.mp4

echo "======================================"
echo " EXP-14 Completed Successfully! "
echo "======================================"
