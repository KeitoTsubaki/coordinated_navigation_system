import torch
from torch import nn

from lidar_network import LiDARNetwork


class CustomActor(nn.Module):
    def __init__(self, state_dim, action_dim, v_max, omega_max):
        super(CustomActor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 256)  # 60 is the output size of LiDARNetwork
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
        self.v_max = v_max
        self.omega_max = omega_max
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, observation):
        # observation is a dictionary from the environment
        # We assume the features are already processed and concatenated
        # observation shape: (batch_size, state_dim)
        
        x = observation
        
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        actions = self.fc3(x)
        
        # The action space is normalized to [-1, 1] for both v and omega.
        # v_n should be in [0, v_max], omega_n in [-omega_max, omega_max].
        # The policy should output actions in [-1, 1] range.
        # Let's assume action[0] is for linear velocity and action[1] for angular.
        v_out = self.sigmoid(actions[:, 0:1]) # Range [0, 1]
        omega_out = self.tanh(actions[:, 1:2]) # Range [-1, 1]
        
        # Scale to the actual action range
        output_v = v_out * self.v_max
        output_omega = omega_out * self.omega_max
        
        return torch.cat([output_v, output_omega], dim=1)
