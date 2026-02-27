import os
import sys
import subprocess
import time
import numpy as np
from stable_baselines3 import PPO

# 基準ディレクトリの設定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
sys.path.append(SCRIPT_DIR)

import config
from src.f1_env import F1TenthRL

def run_command(command, description):
    print(f"\n>>> {description}を実行中...")
    print(f"Command: {command}")
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="")
        process.wait()
        if process.returncode != 0:
            print(f"!!! Error: {description}が失敗しました (Exit code: {process.returncode})")
            return False
        return True
    except Exception as e:
        print(f"!!! Exception: {e}")
        return False

def main():
    print("="*50)
    print("🚀 F1Tenth 改修環境 自動検証スクリプト")
    print("="*50)

    # 1. 観測空間の整合性チェック
    print("\n[1/3] 観測空間の整合性チェック")
    try:
        env = F1TenthRL(config.MAP_PATH)
        obs = env.reset()
        expected_shape = env.observation_space.shape
        print(f"期待される形状: {expected_shape}")
        print(f"実際の観測データの形状: {obs.shape}")
        
        if obs.shape != expected_shape:
            print("!!! Error: 観測データの形状が一致しません")
            sys.exit(1)
        print("✅ 観測空間のチェック完了")
        env.env.close()
    except Exception as e:
        print(f"!!! Error: 環境の初期化中にエラーが発生しました: {e}")
        sys.exit(1)

    # 2. 短時間学習の実行 (2048ステップ)
    # テスト用にconfigを一時的に上書きする代わりに、引数で渡すか小規模な学習を叩く
    print("\n[2/3] 超短時間学習の実行 (動作確認用)")
    test_steps = 2048
    test_model_name = f"test_verify_model"
    test_model_path = os.path.join(config.MODEL_DIR, test_model_name)
    
    # train.py をサブプロセスで実行（ハイパーパラメータを上書き）
    train_cmd = f"python3 {SCRIPT_DIR}/train.py --steps {test_steps} --model {test_model_path}"
    if not run_command(train_cmd, "短時間学習"):
        sys.exit(1)
    
    if not os.path.exists(test_model_path + ".zip"):
        print(f"!!! Error: モデルファイルが生成されませんでした: {test_model_path}.zip")
        sys.exit(1)
    print("✅ 学習プロセスのチェック完了")

    # 3. 描画・GIF生成の実行
    print("\n[3/3] 描画とGIF生成の実行")
    test_mp4_path = os.path.join(config.GIF_DIR, "test_verify_simulation.mp4")
    enjoy_steps = 300
    enjoy_cmd = f"python3 {SCRIPT_DIR}/enjoy_wide.py --steps {enjoy_steps} --model {test_model_path} --save {test_mp4_path}"
    
    if not run_command(enjoy_cmd, "描画テスト"):
        sys.exit(1)
        
    if not os.path.exists(test_mp4_path):
        print(f"!!! Error: GIFファイルが生成されませんでした: {test_mp4_path}")
        sys.exit(1)
    print(f"✅ 描画プロセスのチェック完了 (保存先: {test_mp4_path})")

    print("\n" + "="*50)
    print("✨ すべての検証項目をクリアしました！")
    print("="*50)

if __name__ == "__main__":
    main()
