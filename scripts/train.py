# import gym
# import f110_gym
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# import os
# import config

# class F1TenthRL(gym.Env):
#     def __init__(self, map_path):
#         super(F1TenthRL, self).__init__()
#         # 登録名が 'f110-v0' か 'f110_gym:f110-v0' かは環境依存ですが、
#         # 手動実行で成功した 'f110-v0' に統一します。
#         self.env = gym.make('f110-v0', map=map_path, map_ext='.pgm', num_agents=1)
        
#         # [ハンドル, 速度] の2次元アクション
#         self.action_space = gym.spaces.Box(
#             low=np.array([-1.0, -1.0]), 
#             high=np.array([1.0, 1.0]), 
#             shape=(2,), 
#             dtype=np.float32
#         )
#         self.observation_space = gym.spaces.Box(low=0, high=30, shape=(1080,), dtype=np.float32)

#     def reset(self):
#         initial_poses = np.array([[0.0, 0.0, 0.0]])
#         # 戻り値の数に柔軟に対応
#         result = self.env.reset(poses=initial_poses)
#         if isinstance(result, tuple):
#             obs = result[0]
#         else:
#             obs = result
#         return obs['scans'][0].astype(np.float32)

#     def step(self, action):
#         steer = action[0] * config.STEER_SENSITIVITY
#         # action[1] から速度へ変換
#         speed = config.MIN_SPEED + (action[1] + 1.0) * (config.MAX_SPEED - config.MIN_SPEED) / 2.0
        
#         obs, _, done, info = self.env.step(np.array([[steer, speed]]))
#         scans = obs['scans'][0]
        
#         # info が None や辞書以外の場合に備えて辞書化
#         if info is None: info = {}
        
#         reward = config.calculate_reward(scans, action, done, speed)
        
#         # SB3向けに型を厳密にする
#         return scans.astype(np.float32), float(reward), bool(done), info

# def main():
#     if not os.path.exists(config.MODEL_DIR):
#         os.makedirs(config.MODEL_DIR, exist_ok=True)

#     env = F1TenthRL(config.MAP_PATH)
#     env = DummyVecEnv([lambda: env])

#     model = PPO(
#         "MlpPolicy", 
#         env, 
#         learning_rate=config.LEARNING_RATE,
#         policy_kwargs=dict(net_arch=config.NET_ARCH),
#         verbose=1, 
#         device=config.DEVICE
#     )

#     print(f"--- 可変速度モードで学習開始: {config.MODEL_NAME} ---")
    
#     checkpoint_callback = CheckpointCallback(save_freq=1000000, save_path=config.MODEL_DIR)
#     model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=checkpoint_callback)
#     #model.learn(total_timesteps=config.TOTAL_TIMESTEPS)
#     model.save(config.MODEL_PATH)
#     print(f"--- 保存完了: {config.MODEL_PATH} ---")

# if __name__ == '__main__':
#     main()

import gym
import f110_gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import config

class F1TenthRL(gym.Env):
    def __init__(self, map_path):
        super(F1TenthRL, self).__init__()
        self.env = gym.make('f110-v0', map=map_path, map_ext='.pgm', num_agents=1)
        
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            shape=(2,), 
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(low=0, high=30, shape=(1080,), dtype=np.float32)

    def reset(self):
        initial_poses = np.array([[0.0, 0.0, 0.0]])
        result = self.env.reset(poses=initial_poses)
        obs = result[0] if isinstance(result, tuple) else result
        # そのままのLiDARデータを返す
        return obs['scans'][0].astype(np.float32)

    def step(self, action):
        steer = action[0] * config.STEER_SENSITIVITY
        speed = config.MIN_SPEED + (action[1] + 1.0) * (config.MAX_SPEED - config.MIN_SPEED) / 2.0
        
        obs, _, done, info = self.env.step(np.array([[steer, speed]]))
        
        # --- 変更点：視野制限（スライス）を削除 ---
        scans = obs['scans'][0]
        
        if info is None: info = {}
        reward = config.calculate_reward(scans, action, done, speed)
        
        # SB3向けに型を整える（これはエラー防止のため残す）
        return scans.astype(np.float32), float(reward), bool(done), info

def main():
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR, exist_ok=True)

    # チェックポイント用フォルダ
    checkpoint_dir = os.path.join(config.MODEL_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 100万ステップごとに自動保存
    checkpoint_callback = CheckpointCallback(
        save_freq=1000000, 
        save_path=checkpoint_dir,
        name_prefix=config.MODEL_NAME
    )

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

    print(f"--- 学習開始: {config.MODEL_NAME} ---")
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=checkpoint_callback
    )
    
    model.save(config.MODEL_PATH)
    print(f"--- 完了: {config.MODEL_PATH} ---")

if __name__ == '__main__':
    main()