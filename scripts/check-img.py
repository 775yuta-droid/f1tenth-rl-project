# import gym
# import numpy as np

# # マップを指定して環境を作成
# env = gym.make('f110_gym-v0', 
#                map='/opt/f1tenth_gym/gym/f110_gym/envs/maps/my_map', 
#                map_ext='.png', 
#                num_agents=1)

# # とりあえず [0, 0, 0] でリセットを試みる
# try:
#     obs, reward, done, info = env.reset(np.array([[0.0, 0.0, 0.0]]))
#     print("成功！ [0, 0, 0] は走行可能エリア内です。")
# except:
#     print("失敗！ [0, 0, 0] は壁の中、あるいはコース外です。")

import gym
import f110_gym  # ★これを追加して環境を登録する
import numpy as np

# マップ名は拡張子なしで指定
env = gym.make('f110-v0', 
               map='/opt/f1tenth_gym/gym/f110_gym/envs/maps/my_map', 
               map_ext='.png', 
               num_agents=1)

print("環境の作成に成功しました！")