from gymnasium.envs.registration import register
from .gym_environment import Env
from .gym_environment_square import EnvSquare
from .gym_environment_v2 import EnvV2
from .gym_environment_v3 import EnvV3
from .gym_environment_v4 import EnvV4

register(
    # Renamed for clarity
    id="simulation_env-v1",
    entry_point="rl_training.gym_environment:Env",
)

register(
    id="simulation_env-square-v1",
    entry_point="rl_training.gym_environment_square:EnvSquare",
)

register(
    id="simulation_env-v2",
    entry_point="rl_training.gym_environment_v2:EnvV2",
)

register(
    id="simulation_env-v3",
    entry_point="rl_training.gym_environment_v3:EnvV3",
)

register(
    id="simulation_env-v4",
    entry_point="rl_training.gym_environment_v4:EnvV4",
)

__all__ = ["Env", "EnvSquare", "EnvV2", "EnvV3", "EnvV4"]
