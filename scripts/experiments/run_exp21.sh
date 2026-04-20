#!/bin/bash

# EXP-21 "Frame Stacking & Observation Smoothing"
MODEL_NAME="ppo_10M_exp21_frame_stack"
LOG_FILE="/workspace/logs/exp21_output.log"

echo "=== EXP-21 TRAINING STARTED ===" | tee $LOG_FILE
# Factor 10 (108pts), FrameStack=4, MeanSampling, SteerDelta=0.08
TRAINING_PROFILE=auto python3 scripts/train.py --steps 10000000 --model /workspace/models/$MODEL_NAME 2>&1 | tee -a $LOG_FILE

echo "=== EXP-21 TRAINING FINISHED ===" | tee -a $LOG_FILE
echo "=== EXP-21 EVALUATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/evaluate.py --model $MODEL_NAME --episodes 20 2>&1 | tee -a $LOG_FILE
echo "=== EXP-21 EVALUATION FINISHED ===" | tee -a $LOG_FILE

echo "=== EXP-21 VISUALIZATION STARTED ===" | tee -a $LOG_FILE
python3 scripts/enjoy_wide.py --model $MODEL_NAME --save /workspace/gif/${MODEL_NAME}.mp4 2>&1 | tee -a $LOG_FILE
echo "=== EXP-21 VISUALIZATION FINISHED ===" | tee -a $LOG_FILE
