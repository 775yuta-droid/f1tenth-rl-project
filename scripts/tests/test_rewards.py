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
    """標準的なテスト用設定"""
    return RewardConfig(
        reward_collision=-2000.0,  # 衝突時の報酬
        reward_survival=0.02,      # 生存報酬
        reward_front_weight=3.0,   # 前方報酬の重み
        reward_speed_weight=1.0,   # 速度報酬の重み
        reward_centrality_weight=0.5,  # 中央維持報酬の重み
        reward_distance_weight=1.0,  # 距離報酬の重み
        reward_progress_weight=2.0,  # 進捗報酬の重み
        max_speed=2.5  # 最大速度
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
    """前方距離による報酬の変化を確認 (270-810の範囲)"""
    # 遠くに壁がある場合
    scan_far = np.full(1080, 20.0)
    reward_far = calculate_reward(scan_far, [0.0, 2.0], False, 2.0, reward_config=base_config)
    
    # 近くに壁がある場合 (ただし2m以上)
    scan_near = np.full(1080, 3.0)
    reward_near = calculate_reward(scan_near, [0.0, 2.0], False, 2.0, reward_config=base_config)
    
    assert reward_far > reward_near

def test_centrality_reward(base_config):
    """中央維持報酬の確認 (270付近と810付近)"""
    # 中央にいる場合 (左右均等)
    scan_center = np.full(1080, 5.0)
    reward_center = calculate_reward(scan_center, [0.0, 1.0], False, 1.0, reward_config=base_config)
    
    # 左に寄っている場合 (左が近く、右が遠い)
    scan_left = np.full(1080, 5.0)
    scan_left[800:820] = 1.0  # 左側を近くする
    reward_left = calculate_reward(scan_left, [0.0, 1.0], False, 1.0, reward_config=base_config)
    
    assert reward_center > reward_left

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
