import torch

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.ddpg.policies import MlpPolicy

from actor import CustomActor
from critic import CustomCritic
from lidar_network import LiDARNetwork

class CustomDDPG(MlpPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, v_max=1.2, omega_max=4.71, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        self.v_max = v_max
        self.omega_max = omega_max
        
        discrete_state_dim = 6 # [distance_to_goal, angle_to_goal, v, omega, distance_to_closest_obstacle, angle_to_closest_obstacle]
        lidar_dim = 60
        state_dim = discrete_state_dim + lidar_dim
        action_dim = 2 # [v_n, omega_n]

        self.lidar_network = LiDARNetwork()
        
        self.actor = CustomActor(state_dim, action_dim, v_max, omega_max)
        self.actor_target = CustomActor(state_dim, action_dim, v_max, omega_max)

        self.critic = CustomCritic(state_dim, action_dim)
        self.critic_target = CustomCritic(state_dim, action_dim)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_schedule(1))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_schedule(1))