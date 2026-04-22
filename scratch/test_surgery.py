import gym
import f110_gym
import numpy as np
from f110_gym.envs.laser_models import ScanSimulator2D

def test_surgery():
    # 1. 普通に作成 (1080本になるはず)
    env = gym.make('f110-v0', map='/opt/f1tenth_gym/gym/f110_gym/envs/maps/levine', num_agents=1)
    agent = env.sim.agents[0]
    print(f"Before surgery: {agent.num_beams} beams")

    # 2. 外科手術
    new_num_beams = 1440
    fov = 4.7 # デフォルト
    
    # クラス属性を書き換える
    from f110_gym.envs.base_classes import RaceCar
    base_env = env.unwrapped
    RaceCar.scan_simulator = ScanSimulator2D(new_num_beams, fov)
    RaceCar.scan_simulator.set_map(base_env.map_path, base_env.map_ext)
    RaceCar.cosines = np.cos(np.linspace(-fov/2., fov/2., new_num_beams))
    RaceCar.scan_angles = np.linspace(-fov/2., fov/2., new_num_beams)
    RaceCar.side_distances = np.ones(new_num_beams)

    # インスタンス属性も一応合わせておく
    agent.num_beams = new_num_beams

    # Gymの観測空間も書き換え（存在する場合）
    if hasattr(env, 'observation_space') and env.observation_space is not None:
        from gym import spaces
        env.observation_space.spaces['scans'] = spaces.Box(low=0, high=100, shape=(1, new_num_beams), dtype=np.float32)

    # 3. 確認
    obs = env.reset(poses=np.array([[0, 0, 0]]))
    raw_scans = obs[0]['scans'][0] if isinstance(obs, tuple) else obs['scans'][0]
    print(f"After surgery: {agent.num_beams} beams")
    print(f"Observation size: {len(raw_scans)}")

if __name__ == '__main__':
    test_surgery()
