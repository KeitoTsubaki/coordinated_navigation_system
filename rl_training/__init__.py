from gymnasium.envs.registration import register
from .gym_environment_square import EnvSquare
from .gym_environment_v4 import EnvV4

# 旧バージョン (simulation_env-v1〜v3) は archive/rl_training/ に移動済み

register(
    id="simulation_env-square-v1",
    entry_point="rl_training.gym_environment_square:EnvSquare",
)

register(
    id="simulation_env-v4",
    entry_point="rl_training.gym_environment_v4:EnvV4",
)

__all__ = ["EnvSquare", "EnvV4"]
