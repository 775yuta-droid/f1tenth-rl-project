import gym
import f110_gym
import numpy as np

env = gym.make('f110-v0', map='/opt/f1tenth_gym/gym/f110_gym/envs/maps/levine', num_agents=1, num_beams=1440)
obs = env.reset(poses=np.array([[0, 0, 0]]))
raw_scans = obs[0]['scans'][0] if isinstance(obs, tuple) else obs['scans'][0]
print(f"Number of beams in raw observation: {len(raw_scans)}")
