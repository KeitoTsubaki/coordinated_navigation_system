import torch
from torch import nn

from lidar_network import LiDARNetwork
from utils import split_obs

class CustomCritic(nn.Module):
    def __init__(self, shared_lidar_network, act_dim=2):
        super(CustomCritic, self).__init__()
        
        self.lidar_network = shared_lidar_network
        self.state_net = nn.Sequential(
            nn.Linear(66, 256),  # 60 is the output size of LiDARNetwork
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01)
        )
        self.action_net = nn.Sequential(
            nn.Linear(act_dim, 128),
            nn.LeakyReLU(0.01)
        )
        self.combined_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1)
        )
            
    def forward(self, obs_dict, action):
        # observation is a dictionary from the environment
        # We assume the features are already processed and concatenated
        # observation shape: (batch_size, state_dim)
        # action: [v_n, omega_n]
        
        lidar, discrete_state = split_obs(obs_dict)
        lidar_features = self.lidar_network(lidar)  # (batch_size, 60)
        observation = torch.cat([lidar_features, discrete_state], dim=1) 
        
        state_x = self.state_net(observation)
        
        action_x = self.action_net(action)
        
        combined_x = torch.cat([state_x, action_x], dim=1)
        
        q = self.combined_net(combined_x)
        
        return q
    
    def set_training_mode(self, mode):
        self.train(mode)
        return self