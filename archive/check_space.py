import gymnasium as gym
from gymnasium.spaces import *
import rl_training

ENV_ID = 'simulation_env-v0'

def print_spaces(label, space):
    print(f"{label}: {space}")


env = gym.make(ENV_ID)

print(f"Environment ID: {ENV_ID}")
print_spaces("Observation Space", env.observation_space)
print_spaces("Action Space", env.action_space)