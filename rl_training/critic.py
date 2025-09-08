import torch
from torch import nn

from lidar_network import LiDARNetwork


class CustomCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CustomCritic, self).__init__()
        
        self.state_fc1 = nn.Linear(state_dim, 256)  # 60 is the output size of LiDARNetwork
        self.state_fc2 = nn.Linear(256, 128)
        
        self.action_fc1 = nn.Linear(action_dim, 128)
        
        self.combined_fc1 = nn.Linear(256, 128)
        self.combined_fc2 = nn.Linear(128, 1)
        
        self.leaky_relu = nn.LeakyReLU(0.01)
            
    def forward(self, observation, action):
        # observation is a dictionary from the environment
        # We assume the features are already processed and concatenated
        # observation shape: (batch_size, state_dim)
        # action: [v_n, omega_n]
        
        state_x = observation
        state_x = self.leaky_relu(self.state_fc1(state_x))
        state_x = self.leaky_relu(self.state_fc2(state_x))
        
        action_x = self.leaky_relu(self.action_fc1(action))
        
        combined_x = torch.cat([state_x, action_x], dim=1)
        combined_x = self.leaky_relu(self.combined_fc1(combined_x))
        q_value = self.combined_fc2(combined_x)
        
        return q_value