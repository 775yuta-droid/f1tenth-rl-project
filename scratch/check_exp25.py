from stable_baselines3 import PPO
model = PPO.load("models/ppo_10M_exp25_fast_stable.zip", device="cpu")
print("Policy:", type(model.policy))
print("Observation space:", model.observation_space)
