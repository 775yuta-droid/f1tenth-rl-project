import gym
import f110_gym

env = gym.make('f110-v0', map='/opt/f1tenth_gym/gym/f110_gym/envs/maps/levine', num_agents=1)
# アンラップして中身を見る
base_env = env.unwrapped
print(f"Methods of F110Env: {dir(base_env)}")
