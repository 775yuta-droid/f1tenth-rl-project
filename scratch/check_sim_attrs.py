import gym
import f110_gym
import numpy as np

env = gym.make('f110-v0', map='/opt/f1tenth_gym/gym/f110_gym/envs/maps/levine', num_agents=1)
print("Attributes of env.sim:")
print(dir(env.sim))

if hasattr(env.sim, 'scan_simulator'):
    print("\nAttributes of env.sim.scan_simulator:")
    print(dir(env.sim.scan_simulator))
    print(f"\nCurrent num_beams: {env.sim.scan_simulator.num_beams}")
