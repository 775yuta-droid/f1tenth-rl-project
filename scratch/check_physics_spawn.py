
import gym
import numpy as np
import os
import sys

# Add project root to path
PROJECT_ROOT = "/home/yuta775/projects/f1tenth-rl-project"
sys.path.append(PROJECT_ROOT)

from src import config

def test_spawn_safety():
    map_path = "/home/yuta775/projects/f1tenth-rl-project/my_maps/testmap-tamoku/map-tamoku"
    
    # Simple gym make to avoid our wrapper's complexity first
    env = gym.make('f110-v0', map=map_path, map_ext='.pgm', num_agents=1, timestep=config.SIM_TIMESTEP)
    
    # Apply car dimensions
    env.params['length'] = config.CAR_LENGTH
    env.params['width'] = config.CAR_WIDTH
    
    poses = config.START_POSES
    
    for i, base_pose in enumerate(poses):
        collisions = 0
        trials = 100
        print(f"\nTesting Spawn #{i}: {base_pose}")
        
        for _ in range(trials):
            sx, sy, syaw = base_pose
            sx += np.random.uniform(-0.1, 0.1)
            sy += np.random.uniform(-0.1, 0.1)
            syaw += np.random.uniform(-0.01, 0.01)
            
            obs, _, done, info = env.reset(np.array([[sx, sy, syaw]]))
            
            # Check collision from info or obs
            if done or obs['collisions'][0] > 0:
                collisions += 1
        
        print(f"  Collision Rate: {collisions}/{trials} ({collisions/trials*100:.1f}%)")

if __name__ == "__main__":
    test_spawn_safety()
