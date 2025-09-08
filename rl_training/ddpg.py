import torch
from torch import nn
from typing import Any, Dict, List, Optional, Type

import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule

from actor import CustomActor
from critic import CustomCritic
from lidar_network import LiDARNetwork


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Dict):
        # We must specify the features_dim, this is the dimension of the output of the extractor
        discrete_state_dim = 2 + 2 + 2 # goal, velocity, closest_obstacle
        lidar_dim = 60
        features_dim = discrete_state_dim + lidar_dim
        super().__init__(observation_space, features_dim)

        self.lidar_network = LiDARNetwork()

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process LiDAR data
        lidar_features = self.lidar_network(observations["lidar"])

        # Concatenate with other observation parts
        other_features = torch.cat([
            observations["goal"],
            observations["velocity"],
            observations["closest_obstacle"]
        ], dim=1)

        # Combine all features
        return torch.cat([lidar_features, other_features], dim=1)


class CustomDDPGPolicy(BasePolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        v_max: float = 1.2,
        omega_max: float = 4.71,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, **kwargs, features_extractor_class=CustomCombinedExtractor)
        
        state_dim = self.features_dim
        action_dim = action_space.shape[-1]

        self.actor = CustomActor(state_dim, action_dim, v_max, omega_max)
        self.actor_target = CustomActor(state_dim, action_dim, v_max, omega_max)
        self.critic = CustomCritic(state_dim, action_dim)
        self.critic_target = CustomCritic(state_dim, action_dim)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # This method is not used by DDPG for training, but can be for prediction
        # The actual forward pass is handled by the actor and critic directly in the DDPG algorithm
        features = self.extract_features(obs)
        return self.actor(features)
