from gymnasium.envs.registration import register
from matplotllib_2d_simulator.gym_environment import Env

register(
    id='matplotllib_2d_simulator-v0',
    entry_point='matplotllib_2d_simulator.gym_environment:Env',
)