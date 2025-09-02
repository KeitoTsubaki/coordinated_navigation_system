from gymnasium.envs.registration import register
from .gym_environment import Env

register(
    # Renamed for clarity
    id='simulation_env-v0',
    entry_point='rl_training.gym_environment:Env',
)