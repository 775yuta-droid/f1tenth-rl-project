import gym
import f110_gym

env = gym.make('f110-v0', map='/opt/f1tenth_gym/gym/f110_gym/envs/maps/levine', num_agents=1)
print(f"Has observation_space: {hasattr(env, 'observation_space')}")
if hasattr(env, 'observation_space'):
    print(f"observation_space: {env.observation_space}")

obs = env.reset()
print(f"\nType of obs: {type(obs)}")
if isinstance(obs, tuple):
    print("Obs is a tuple (standard in newer Gym/Gymnasium)")
    obs = obs[0]

if isinstance(obs, dict):
    print(f"Obs keys: {obs.keys()}")
    if 'scans' in obs:
        print(f"Scans shape: {obs['scans'].shape}")
