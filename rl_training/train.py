import gymnasium as gym
from stable_baselines3 import DDPG
from rl_training.ddpg import CustomDDPGPolicy
from rl_training.gym_environment import Env

env = gym.make('simulation_env-v0', render_mode='rgb_array')

model = DDPG(
    policy=CustomDDPGPolicy,
    env=env,
    verbose=1,
)

model.learn(total_timesteps=100000)

model.save("ddpg_model")

del model