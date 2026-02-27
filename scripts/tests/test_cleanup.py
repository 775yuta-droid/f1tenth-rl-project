#!/usr/bin/env python3
"""
クリーンアップ後のコードをテストするスクリプト
"""
import sys
import os

# パスの設定 (scripts/tests/ から2階層上がプロジェクトルート)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

print("=" * 50)
print("クリーンアップ後のコードテスト")
print("=" * 50)

# テスト1: configモジュールのimport
print("\n[テスト1] configモジュールのimport...")
try:
    sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
    import config
    print(f"  ✓ 成功")
    print(f"    - DEVICE: {config.DEVICE}")
    print(f"    - TOTAL_TIMESTEPS: {config.TOTAL_TIMESTEPS}")
    print(f"    - MAP_PATH: {config.MAP_PATH}")
except Exception as e:
    print(f"  ✗ 失敗: {e}")
    sys.exit(1)

# テスト2: 環境クラスのimport
print("\n[テスト2] F1TenthRL環境クラスのimport...")
try:
    from src.f1_env import F1TenthRL
    print(f"  ✓ 成功")
except Exception as e:
    print(f"  ✗ 失敗: {e}")
    sys.exit(1)

# テスト3: 環境の初期化
print("\n[テスト3] 環境の初期化...")
try:
    env = F1TenthRL(config.MAP_PATH)
    print(f"  ✓ 成功")
    print(f"    - アクション空間: {env.action_space}")
    print(f"    - 観測空間: {env.observation_space}")
except Exception as e:
    print(f"  ✗ 失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# テスト4: 環境のリセット
print("\n[テスト4] 環境のリセット...")
try:
    obs = env.reset()
    print(f"  ✓ 成功")
    print(f"    - 観測の形状: {obs.shape}")
    print(f"    - 観測の型: {obs.dtype}")
except Exception as e:
    print(f"  ✗ 失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# テスト5: ステップ実行
print("\n[テスト5] ステップ実行...")
try:
    import numpy as np
    action = np.array([0.0, 0.5])  # 直進、中速
    obs, reward, done, info = env.step(action)
    print(f"  ✓ 成功")
    print(f"    - 報酬: {reward:.4f}")
    print(f"    - 終了フラグ: {done}")
    print(f"    - 観測の形状: {obs.shape}")
except Exception as e:
    print(f"  ✗ 失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# テスト6: 報酬関数のテスト
print("\n[テスト6] 報酬関数のテスト...")
try:
    scans = np.ones(1080) * 10.0  # 全方向10m
    action = np.array([0.0, 0.5])
    reward = config.calculate_reward(scans, action, False, 2.0)
    print(f"  ✓ 成功")
    print(f"    - 通常時の報酬: {reward:.4f}")
    
    reward_collision = config.calculate_reward(scans, action, True, 2.0)
    print(f"    - 衝突時の報酬: {reward_collision:.4f}")
except Exception as e:
    print(f"  ✗ 失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("すべてのテストが成功しました！ ✓")
print("=" * 50)
