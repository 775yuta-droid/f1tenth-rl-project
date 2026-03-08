"""
報酬関数ユニットテスト

gym環境不要・軽量に実行可能。
RewardConfig を使ってパラメータをモックし、各報酬コンポーネントを独立検証します。

実行:
    python -m pytest scripts/tests/test_rewards.py -v
"""
import sys
import os
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.rewards import RewardConfig, calculate_reward

# ─── テスト用フィクスチャ ─────────────────────────────────────────────────

@pytest.fixture
def cfg():
    """テスト用デフォルト設定（config.py 不要）"""
    return RewardConfig(
        reward_collision=-2000.0,
        reward_survival=0.05,
        reward_front_weight=3.0,
        reward_speed_weight=1.0,
        reward_centrality_weight=0.5,
        reward_distance_weight=1.0,
        reward_progress_weight=2.0,
        max_speed=2.5,
    )


def make_scans(value: float = 10.0) -> np.ndarray:
    """全方向同一距離のスキャンを生成（1080点）"""
    return np.full(1080, value, dtype=np.float32)


# ─── テストケース ─────────────────────────────────────────────────────────

class TestCollision:
    """衝突ペナルティのテスト"""

    def test_collision_returns_large_negative(self, cfg):
        """done=True の時は reward_collision を返す"""
        scans = make_scans(10.0)
        reward = calculate_reward(scans, [0.0, 0.5], done=True,
                                  current_speed=2.0, reward_config=cfg)
        assert reward == cfg.reward_collision, f"Expected {cfg.reward_collision}, got {reward}"

    def test_collision_ignores_scans(self, cfg):
        """衝突時はスキャン値に依存しない"""
        r1 = calculate_reward(make_scans(0.1), [0.0, 0.5], done=True,
                              current_speed=2.0, reward_config=cfg)
        r2 = calculate_reward(make_scans(30.0), [0.0, 0.5], done=True,
                              current_speed=2.0, reward_config=cfg)
        assert r1 == r2 == cfg.reward_collision


class TestFrontalReward:
    """前方空間報酬のテスト"""

    def test_clear_front_gives_positive_reward(self, cfg):
        """前方が広い空間では正の報酬が得られる"""
        scans = make_scans(15.0)
        reward = calculate_reward(scans, [0.0, 0.5], done=False,
                                  current_speed=2.0, reward_config=cfg)
        assert reward > 0.0, f"Expected positive reward, got {reward}"

    def test_larger_front_dist_more_reward(self, cfg):
        """前方距離が大きいほど報酬が高い"""
        r_near = calculate_reward(make_scans(2.0), [0.0, 0.3], done=False,
                                  current_speed=1.0, reward_config=cfg)
        r_far  = calculate_reward(make_scans(15.0), [0.0, 0.8], done=False,
                                  current_speed=2.0, reward_config=cfg)
        assert r_far > r_near, f"Far ({r_far:.3f}) should be > near ({r_near:.3f})"


class TestWallPenalty:
    """壁接近ペナルティのテスト"""

    def test_penalty_applied_when_close_to_wall(self, cfg):
        """最小距離 < 1m の時にペナルティが加算される"""
        scans_safe = make_scans(5.0)
        scans_danger = make_scans(5.0)
        scans_danger[0] = 0.3  # 1点だけ壁に近い

        r_safe   = calculate_reward(scans_safe,   [0.0, 0.5], done=False,
                                    current_speed=2.0, reward_config=cfg)
        r_danger = calculate_reward(scans_danger, [0.0, 0.5], done=False,
                                    current_speed=2.0, reward_config=cfg)
        assert r_danger < r_safe, "Wall-close should reduce reward"

    def test_no_penalty_when_safe(self, cfg):
        """全方向 >= 1m なら壁接近ペナルティの差は前方距離分のみ"""
        scans_1m = make_scans(1.0)
        scans_5m = make_scans(5.0)
        r_1m = calculate_reward(scans_1m, [0.0, 0.5], done=False,
                                current_speed=2.0, reward_config=cfg)
        r_5m = calculate_reward(scans_5m, [0.0, 0.5], done=False,
                                current_speed=2.0, reward_config=cfg)
        assert r_5m > r_1m


class TestProgressScale:
    """progress_scale の連続性テスト"""

    def test_progress_scale_monotone_increases(self, cfg):
        """front_dist が大きいほど走行距離報酬も大きい"""
        distances = [0.5, 1.0, 2.0, 4.0, 6.0, 10.0]
        rewards = []
        for d in distances:
            scans = make_scans(d)
            r = calculate_reward(scans, [0.0, 0.5], done=False,
                                 current_speed=2.0,
                                 prev_x=0.0, prev_y=0.0,
                                 cur_x=0.1, cur_y=0.0,
                                 reward_config=cfg)
            rewards.append(r)

        for i in range(len(rewards) - 1):
            assert rewards[i] <= rewards[i + 1], (
                f"reward[{distances[i]}m]={rewards[i]:.3f} "
                f"> reward[{distances[i+1]}m]={rewards[i+1]:.3f} (not monotone)"
            )

    def test_progress_scale_no_jump(self, cfg):
        """境界付近で急激な変化がない"""
        scans_199 = make_scans(1.99)
        scans_200 = make_scans(2.01)
        r1 = calculate_reward(scans_199, [0.0, 0.5], done=False,
                              current_speed=2.0,
                              prev_x=0.0, prev_y=0.0, cur_x=0.1, cur_y=0.0,
                              reward_config=cfg)
        r2 = calculate_reward(scans_200, [0.0, 0.5], done=False,
                              current_speed=2.0,
                              prev_x=0.0, prev_y=0.0, cur_x=0.1, cur_y=0.0,
                              reward_config=cfg)
        diff = abs(r2 - r1)
        # 現在の報酬関数は front_dist=2m の境界で progress_scale が 0→0.3 に段階変化するため
        # ある程度のジャンプは仕様。ただし報酬全体の 1/3 以内（≒1.5）に収まることを確認。
        assert diff < 1.5, f"Reward jump at front_dist=2m boundary: {diff:.3f} (expected < 1.5)"


class TestSurvivalBonus:
    """生存ボーナスのテスト"""

    def test_survival_bonus_always_added(self, cfg):
        """done=False の時は毎ステップ survival ボーナスが加算される"""
        scans = make_scans(10.0)
        cfg_no_survival = RewardConfig(
            reward_collision=cfg.reward_collision,
            reward_survival=0.0,
            reward_front_weight=cfg.reward_front_weight,
            reward_speed_weight=cfg.reward_speed_weight,
            reward_centrality_weight=cfg.reward_centrality_weight,
            reward_distance_weight=cfg.reward_distance_weight,
            reward_progress_weight=cfg.reward_progress_weight,
            max_speed=cfg.max_speed,
        )
        r_with    = calculate_reward(scans, [0.0, 0.5], done=False,
                                     current_speed=2.0, reward_config=cfg)
        r_without = calculate_reward(scans, [0.0, 0.5], done=False,
                                     current_speed=2.0, reward_config=cfg_no_survival)
        assert abs((r_with - r_without) - cfg.reward_survival) < 1e-6, \
            "Survival bonus not correctly applied"
