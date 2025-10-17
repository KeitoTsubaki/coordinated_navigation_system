from gymnasium.envs.registration import register
from .gym_environment import Env
from .gym_environment_v2 import EnvV2

register(
    # Renamed for clarity
    id="simulation_env-v0",
    entry_point="rl_training.gym_environment:Env",
)

register(
    id="simulation_env-v1",
    entry_point="rl_training.gym_environment_v2:EnvV2",
)

__all__ = ["Env", "EnvV2"]
