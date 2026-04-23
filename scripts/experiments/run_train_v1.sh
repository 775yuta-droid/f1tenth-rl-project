#!/bin/bash
# F1TENTH RL Training Script - Optimized for CPU and Performance
# Created: 2026-04-23

# --- 設定 (デフォルト値) ---
MODEL_NAME=${1:-"ppo_hokuyo_270deg_v1"}
STEPS=${2:-10000000}
THREADS=${3:-2}  # ベンチマーク結果に基づく最適値
PROFILE=${4:-"laptop"}

LOG_FILE="train_${MODEL_NAME}.log"
CONTAINER_NAME="f1-sim-latest"

# --- コンテナ実行確認 ---
if ! docker compose ps | grep -q "Up"; then
    echo "❌ エラー: Dockerコンテナが起動していません。"
    echo "docker compose up -d を実行してから再試行してください。"
    exit 1
fi

echo "=========================================================="
echo "🚀 F1TENTH 学習開始: ${MODEL_NAME}"
echo "   ステップ数: ${STEPS}"
echo "   スレッド数: ${THREADS} (最適化済み)"
echo "   プロファイル: ${PROFILE}"
echo "   ログ出力先: ${LOG_FILE}"
echo "=========================================================="

# docker compose exec でバックグラウンド実行
# TORCH_NUM_THREADS を渡して実行
docker compose exec -d ${CONTAINER_NAME} bash -c "
    export TORCH_NUM_THREADS=${THREADS}
    export TRAINING_PROFILE=${PROFILE}
    python3 scripts/train.py --steps ${STEPS} --model /workspace/models/${MODEL_NAME} > /workspace/${LOG_FILE} 2>&1
"

echo "✅ 学習プロセスをバックグラウンドで開始しました。"
echo "📊 進捗を確認するには以下のコマンドを実行してください:"
echo "docker compose exec ${CONTAINER_NAME} tail -f /workspace/${LOG_FILE}"
echo "=========================================================="
