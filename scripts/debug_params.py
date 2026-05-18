
import gym
import f110_gym
import numpy as np
import os
from src import config

def main():
    env = gym.make('f110-v0', map=config.MAP_PATH, map_ext='.pgm', num_agents=1)
    
    # Force params as in f1_env.py
    print(f"Applying config dimensions: {config.CAR_LENGTH} x {config.CAR_WIDTH}")
    env.params['length'] = config.CAR_LENGTH
    env.params['width'] = config.CAR_WIDTH
    if hasattr(env, 'sim') and len(env.sim.agents) > 0:
        env.sim.agents[0].params['length'] = config.CAR_LENGTH
        env.sim.agents[0].params['width'] = config.CAR_WIDTH

    print("\n--- Environment Parameters ---")
    for key, value in env.params.items():
        if key in ['length', 'width']:
            print(f"{key}: {value}")
    
    print("\n--- Agent 0 Internal Parameters ---")
    if hasattr(env, 'sim') and len(env.sim.agents) > 0:
        agent_params = env.sim.agents[0].params
        for key in ['length', 'width', 'lf', 'lr']:
            print(f"{key}: {agent_params.get(key, 'N/A')}")
    
    # Check LiDAR position
    # The default LiDAR offset is often defined in the simulator's agent class
    # but we can try to find it.
    
    obs = env.reset(poses=np.array([[0, 0, 0]]))
    # In 'f110-v0', the state usually contains info about the car's bounding box
    
    print("\nObservation keys:", obs.keys() if isinstance(obs, dict) else "Not a dict")
    
    env.close()

if __name__ == "__main__":
    main()

    