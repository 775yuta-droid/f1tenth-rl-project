#!/bin/bash
# F1TENTH RL Training Script - High Resolution LiDAR (270deg/1080pts)
# Created: 2026-04-23

MODEL_NAME="ppo_hokuyo_270deg_v1"
STEPS=10000000
LOG_FILE="train_${MODEL_NAME}.log"

echo "=========================================================="
echo "🚀 F1TENTH 学習開始: ${MODEL_NAME}"
echo "   ステップ数: ${STEPS}"
echo "   設定環境数: 1 (共有メモリ制限回避のため)"
echo "   ログ出力先: ${LOG_FILE}"
echo "=========================================================="

# docker compose exec でバックグラウンド実行
# -d オプションを使用してデタッチし、出力をコンテナ内のログファイルに保存
docker compose exec -d f1-sim-latest bash -c "python3 scripts/train.py --steps ${STEPS} --model /workspace/models/${MODEL_NAME} > /workspace/${LOG_FILE} 2>&1"

echo "学習プロセスをバックグラウンドで開始しました。"
echo "進捗を確認するには以下のコマンドを実行してください:"
echo "docker compose exec f1-sim-latest tail -f /workspace/${LOG_FILE}"
