import torch
from torch import nn

from lidar_network import LiDARNetwork
from utils import split_obs



class CustomActor(nn.Module):
    def __init__(self, shared_lidar_network, act_dim=2):
        super(CustomActor, self).__init__()
        
        self.lidar_network = shared_lidar_network
        
        self.mlp = nn.Sequential(
            nn.Linear(66, 256),  # 60 is the output size of LiDARNetwork
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01)
        )
        self.head_v = nn.Linear(128, 1)  # Linear velocity output
        self.head_omega = nn.Linear(128, 1)  # Angular velocity output
        
        # self.v_max = v_max
        # self.omega_max = omega_max
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
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
        v_out = self.sigmoid(v) # Range [0, 1]
        omega_out = self.tanh(omega) # Range [-1, 1]
        
        # Scale to the actual action range
        # output_v = v_out * self.v_max
        # output_omega = omega_out * self.omega_max
        
        return torch.cat([v_out, omega_out], dim=1)
    
    def set_training_mode(self, mode):
        self.train(mode)
        return self
