import gym
import f110_gym
import os

env = gym.make('f110-v0', map='/opt/f1tenth_gym/gym/f110_gym/envs/maps/levine', map_ext='.pgm', num_agents=1)
print("Keys in agent.params:")
print(env.sim.agents[0].params.keys())
