import gym
import f110_gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import config

class F1TenthRL(gym.Env):
    def __init__(self, map_path):
        super(F1TenthRL, self).__init__()
        self.env = gym.make('f110_gym:f110-v0', map=map_path, num_agents=1)
        
        # 【修正ポイント】アクションを2次元 [ハンドル, 速度] に設定
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            shape=(2,), 
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(low=0, high=30, shape=(1080,), dtype=np.float32)

    def reset(self):
        obs, reward, done, info = self.env.reset(np.array([[0.0, 0.0, 0.0]]))
        return obs['scans'][0].astype(np.float32)

    def step(self, action):
        # 1. AIの出力(action)を実際の制御値に変換
        steer = action[0] * config.STEER_SENSITIVITY
        
        # action[1] (-1 to 1) を実際の速度に変換
        speed = config.MIN_SPEED + (action[1] + 1.0) * (config.MAX_SPEED - config.MIN_SPEED) / 2.0
        
        # 2. シミュレータを実行
        obs, _, done, info = self.env.step(np.array([[steer, speed]]))
        scans = obs['scans'][0]
        
        # 3. 【ここが重要】報酬計算に speed を追加して渡す
        reward = config.calculate_reward(scans, action, done, speed)
        
        return scans.astype(np.float32), reward, done, info

def main():
    map_path = config.MAP_PATH
    
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)

    # 環境のセットアップ
    env = F1TenthRL(map_path)
    env = DummyVecEnv([lambda: env])

    # モデルの定義
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=config.LEARNING_RATE,
        policy_kwargs=dict(net_arch=config.NET_ARCH),
        verbose=1, 
        device=config.DEVICE
    )

    print(f"--- 可変速度モードで学習開始: {config.MAP_PATH} ---")
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS)

    # 保存
    model.save(config.MODEL_PATH)
    print(f"--- モデルを保存しました: {config.MODEL_PATH} ---")

if __name__ == '__main__':
    main()