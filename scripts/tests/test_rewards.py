import sys
import os
import numpy as np
import pytest

# プロジェクトルートをパスに追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.rewards import calculate_reward, RewardConfig

@pytest.fixture
def base_config():
    """標準的なテスト用設定 (config.py の実際の値に同期)"""
    return RewardConfig(
        reward_collision=-1000.0,   # config.REWARD_COLLISION
        reward_survival=0.05,       # config.REWARD_SURVIVAL
        reward_front_weight=3.0,    # config.REWARD_FRONT_WEIGHT
        reward_speed_weight=1.0,    # config.REWARD_SPEED_WEIGHT
        reward_safety_weight=0.8,   # config.REWARD_SAFETY_WEIGHT (旧 centrality+distance)
        reward_distance_weight=1.0, # config.REWARD_DISTANCE_WEIGHT (互換用)
        reward_progress_weight=1.0, # config.REWARD_PROGRESS_WEIGHT
        max_speed=2.5               # config.MAX_SPEED
    )

@pytest.fixture
def empty_scan():
    """壁が全くない状態のLiDARデータ（1080点）"""
    return np.full(1080, 30.0)

def test_collision_reward(base_config, empty_scan):
    """衝突時の報酬を確認"""
    reward = calculate_reward(empty_scan, [0.0, 1.0], True, 1.0, reward_config=base_config)
    assert reward == base_config.reward_collision

def test_speed_reward_logic(base_config, empty_scan):
    """速度による報酬の変化を確認"""
    # 低速時
    reward_low = calculate_reward(empty_scan, [0.0, 1.0], False, 1.0, reward_config=base_config)
    # 高速時
    reward_high = calculate_reward(empty_scan, [0.0, 2.0], False, 2.0, reward_config=base_config)
    
    assert reward_high > reward_low

def test_front_distance_reward(base_config):
    """前方距離による報酬の変化を確認"""
    # 遠くに壁がある場合
    scan_far = np.full(1080, 20.0)
    reward_far = calculate_reward(scan_far, [0.0, 2.0], False, 2.0, reward_config=base_config)
    
    # 近くに壁がある場合 (ただし2m以上)
    scan_near = np.full(1080, 3.0)
    reward_near = calculate_reward(scan_near, [0.0, 2.0], False, 2.0, reward_config=base_config)
    
    assert reward_far > reward_near

def test_safety_score_monotone(base_config):
    """安全スコア曲線が壁距離に対して単調増加であることを確認 (旧 clearance_reward テスト)"""
    # 壁から十分に離れている場合
    scan_far = np.full(1080, 5.0)
    reward_far = calculate_reward(scan_far, [0.0, 1.0], False, 1.0, reward_config=base_config)

    # 壁に近い場合（最小距離が0.5m）
    scan_near = np.full(1080, 5.0)
    scan_near[0] = 0.5
    reward_near = calculate_reward(scan_near, [0.0, 1.0], False, 1.0, reward_config=base_config)

    # 壁から十分遠い場合のほうが安全スコア報酬が高くなるはず
    assert reward_far > reward_near

def test_safety_score_penalty_close(base_config):
    """壁に0.5m以内まで近づいた時にペナルティが発生することを確認"""
    # 全方向に余裕がある状態
    scan_clear = np.full(1080, 2.0)  # safety_score = 1.0 → (+0.5) * weight
    reward_clear = calculate_reward(scan_clear, [0.0, 1.0], False, 1.0, reward_config=base_config)

    # 全方向が0.5m以下（最悪ケース）
    scan_veryclose = np.full(1080, 0.1)  # safety_score ≈ 0.05 → (-0.45) * weight
    reward_close = calculate_reward(scan_veryclose, [0.0, 1.0], False, 1.0, reward_config=base_config)

    # 近い場合は報酬が低い（ペナルティ的）であることを確認
    assert reward_clear > reward_close

def test_progress_scale_non_zero(base_config):
    """前方至近距離でも progress_scale が0にならないことを確認"""
    scan_very_near = np.full(1080, 1.0) # front_dist < 2.0
    # 進捗がある場合
    reward_with_move = calculate_reward(scan_very_near, [0.0, 1.0], False, 1.0, 
                                        prev_x=0.0, prev_y=0.0, cur_x=1.0, cur_y=0.0, 
                                        reward_config=base_config)
    # 進捗がない場合
    reward_no_move = calculate_reward(scan_very_near, [0.0, 1.0], False, 1.0, 
                                      prev_x=0.0, prev_y=0.0, cur_x=0.0, cur_y=0.0, 
                                      reward_config=base_config)
    
    # progress_scale が 0.1 ならば、移動した方が報酬が高くなるはず
    assert reward_with_move > reward_no_move

def test_steering_stability(base_config, empty_scan):
    """直進時のステアリング安定ボーナスを確認"""
    # ステアリング 0
    reward_straight = calculate_reward(empty_scan, [0.0, 1.0], False, 1.0, reward_config=base_config)
    # ステアリングを大きく切る
    reward_steer = calculate_reward(empty_scan, [1.0, 1.0], False, 1.0, reward_config=base_config)
    
    assert reward_straight > reward_steer
