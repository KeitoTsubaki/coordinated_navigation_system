import torch
from torch import nn
from stable_baselines3.common.policies import BasePolicy

from lidar_network import LiDARNetwork
from utils import split_obs



class CustomActor(BasePolicy):
    def __init__(self, observation_space, action_space, shared_lidar_network, lr_schedule, v_max, omega_max):
        super().__init__(observation_space, action_space, lr_schedule, squash_output=False)
        # squash_output=False: 自動的にActionがtanhで正規化されるのを防ぐ。今回の場合はtanhとsigmoidを使用するため。
        
        self.lidar_network = shared_lidar_network
        
        self.mlp = nn.Sequential(
            nn.Linear(66, 256),  # 60 is the output size of LiDARNetwork
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01)
        )
        self.head_v = nn.Linear(128, 1)  # Linear velocity output
        self.head_omega = nn.Linear(128, 1)  # Angular velocity output
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.v_max = v_max
        self.omega_max = omega_max
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1))
        
    def forward(self, obs_dict):
        # observation is a dictionary from the environment
        # We assume the features are already processed and concatenated
        # observation shape: (batch_size, state_dim)
        lidar, discrete_state = split_obs(obs_dict)
        lidar_features = self.lidar_network(lidar)  # (batch_size, 60)
        observation = torch.cat([lidar_features, discrete_state], dim=1) 
        
        x = self.mlp(observation)
        v = self.head_v(x)
        omega = self.head_omega(x)
        
        # The action space is normalized to [-1, 1] for both v and omega.
        # v_n should be in [0, v_max], omega_n in [-omega_max, omega_max].
        # The policy should output actions in [-1, 1] range.
        v_out = self.sigmoid(v) * self.v_max # Range [0, 1] * v_max = [0, v_max]
        omega_out = self.tanh(omega) * self.omega_max # Range [-1, 1] * omega_max = [-omega_max, omega_max]
        
        # Scale to the actual action range
        # output_v = v_out * self.v_max
        # output_omega = omega_out * self.omega_max
        
        return torch.cat([v_out, omega_out], dim=1)
    
    def _predict(self, observation, deterministic = True):
        # deterministic: Falseにすると、探索ノイズが入る。学習中や行動サンプリング時に使われる。Trueはノイズが入らず、予測時に使われる。
        return self.forward(observation)
