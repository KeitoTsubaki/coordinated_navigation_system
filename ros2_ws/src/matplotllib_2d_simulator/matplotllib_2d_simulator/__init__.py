from gymnasium.envs.registration import register

register(
    id='matplotllib_2d_simulator-v0',
    entry_point='matplotllib_2d_simulator.gym_environment:Env',
)