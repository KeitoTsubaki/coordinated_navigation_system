import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from stable_baselines3 import DDPG
# __init__.pyで環境が登録されるため、rl_trainingをインポートします。
import rl_training
from ddpg import CustomDDPGPolicy

env = gym.make('simulation_env-v0', render_mode='human')

# ポリシーに渡す引数を準備します。
policy_kwargs = {
    "v_max": env.unwrapped.v_max,
    "omega_max": env.unwrapped.omega_max,
}

model = DDPG(
    policy=CustomDDPGPolicy, # 文字列 'CustomDDPGPolicy' ではなくクラスを渡します
    env=env,
    policy_kwargs=policy_kwargs,
    verbose=1,
)

model.learn(total_timesteps=100000)

model.save("ddpg_model")

del model