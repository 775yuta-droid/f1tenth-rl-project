# import gym
# #import gymnasium as gym
# import f110_gym
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# import os
# import config

# class F1TenthRL(gym.Env):
#     def __init__(self, map_path):
#         super(F1TenthRL, self).__init__()
#         self.env = gym.make('f110-v0', map=map_path, num_agents=1)
        
#         # 【修正ポイント】アクションを2次元 [ハンドル, 速度] に設定
#         self.action_space = gym.spaces.Box(
#             low=np.array([-1.0, -1.0]), 
#             high=np.array([1.0, 1.0]), 
#             shape=(2,), 
#             dtype=np.float32
#         )
#         self.observation_space = gym.spaces.Box(low=0, high=30, shape=(1080,), dtype=np.float32)

#     def reset(self):
#         initial_poses = np.array([[0.0, 0.0, 0.0]])
#         obs, reward, done, info = self.env.reset(poses = initial_poses)
#         return obs['scans'][0].astype(np.float32)

#     def step(self, action):
#         # 1. AIの出力(action)を実際の制御値に変換
#         steer = action[0] * config.STEER_SENSITIVITY
        
#         # action[1] (-1 to 1) を実際の速度に変換
#         speed = config.MIN_SPEED + (action[1] + 1.0) * (config.MAX_SPEED - config.MIN_SPEED) / 2.0
        
#         # 2. シミュレータを実行
#         obs, _, done, info = self.env.step(np.array([[steer, speed]]))
#         scans = obs['scans'][0]
        
#         # 3. 【ここが重要】報酬計算に speed を追加して渡す
#         reward = config.calculate_reward(scans, action, done, speed)
        
#         return scans.astype(np.float32), reward, done, info

# def main():
#     map_path = config.MAP_PATH
    
#     if not os.path.exists(config.MODEL_DIR):
#         os.makedirs(config.MODEL_DIR)

#     # 環境のセットアップ
#     env = F1TenthRL(map_path)
#     env = DummyVecEnv([lambda: env])

#     # モデルの定義
#     model = PPO(
#         "MlpPolicy", 
#         env, 
#         learning_rate=config.LEARNING_RATE,
#         policy_kwargs=dict(net_arch=config.NET_ARCH),
#         verbose=1, 
#         device=config.DEVICE
#     )

#     print(f"--- 可変速度モードで学習開始: {config.MAP_PATH} ---")
#     model.learn(total_timesteps=config.TOTAL_TIMESTEPS)

#     # 保存
#     model.save(config.MODEL_PATH)
#     print(f"--- モデルを保存しました: {config.MODEL_PATH} ---")

# if __name__ == '__main__':
#     main()

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
        # 登録名が 'f110-v0' か 'f110_gym:f110-v0' かは環境依存ですが、
        # 手動実行で成功した 'f110-v0' に統一します。
        self.env = gym.make('f110-v0', map=map_path, num_agents=1)
        
        # [ハンドル, 速度] の2次元アクション
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            shape=(2,), 
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(low=0, high=30, shape=(1080,), dtype=np.float32)

    def reset(self):
        initial_poses = np.array([[0.0, 0.0, 0.0]])
        # 戻り値の数に柔軟に対応
        result = self.env.reset(poses=initial_poses)
        if isinstance(result, tuple):
            obs = result[0]
        else:
            obs = result
        return obs['scans'][0].astype(np.float32)

    def step(self, action):
        steer = action[0] * config.STEER_SENSITIVITY
        # action[1] から速度へ変換
        speed = config.MIN_SPEED + (action[1] + 1.0) * (config.MAX_SPEED - config.MIN_SPEED) / 2.0
        
        obs, _, done, info = self.env.step(np.array([[steer, speed]]))
        scans = obs['scans'][0]
        
        # info が None や辞書以外の場合に備えて辞書化
        if info is None: info = {}
        
        reward = config.calculate_reward(scans, action, done, speed)
        
        # SB3向けに型を厳密にする
        return scans.astype(np.float32), float(reward), bool(done), info

def main():
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR, exist_ok=True)

    env = F1TenthRL(config.MAP_PATH)
    env = DummyVecEnv([lambda: env])

    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=config.LEARNING_RATE,
        policy_kwargs=dict(net_arch=config.NET_ARCH),
        verbose=1, 
        device=config.DEVICE
    )

    print(f"--- 可変速度モードで学習開始: {config.MODEL_NAME} ---")
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS)
    model.save(config.MODEL_PATH)
    print(f"--- 保存完了: {config.MODEL_PATH} ---")

if __name__ == '__main__':
    main()