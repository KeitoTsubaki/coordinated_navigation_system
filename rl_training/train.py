import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
from stable_baselines3 import DDPG

# __init__.pyで環境が登録されるため、rl_trainingをインポートします。
import rl_training
from rl_training.custom_ddpg_policy import CustomDDPGPolicy

env = gym.make("simulation_env-v0", render_mode="human")

model = DDPG(
    policy=CustomDDPGPolicy,  # 文字列 'CustomDDPGPolicy' ではなくクラスを渡します
    env=env,
    learning_rate=1e-3,
    buffer_size=100_000,
    batch_size=64,
    tau=0.005,
    gamma=0.99,
    verbose=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

model.learn(total_timesteps=100_000)

model.save("ddpg_model")

del model
