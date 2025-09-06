import torch
from torch import nn
from lidar_network import LiDARNetwork

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, v_max, omega_max):
        super(Actor, self).__init__()
        
        self.lidar_network = LiDARNetwork()
        
        self.fc1 = nn.Linear(state_dim + 60, 256)  # 60 is the output size of LiDARNetwork
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
        self.v_max = v_max
        self.omega_max = omega_max
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def forward(self, state, lidar_data):
        # state shape: (batch_size, state_dim)
        # lidar_data shape: (batch_size, 4, 460)
        
        lidar_feature = self.lidar_network(lidar_data)  # (batch_size, 60)
        
        x = torch.cat([lidar_feature, state], dim=1)  # (batch_size, 60 + state_dim)
        
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        
        v_n = self.sigmoid(x)  # Output in range [-1, 1]
        omega_n = self.tanh(x)  # Output in range [0, 1]
        
        output_v_n = v_n * self.v_max
        output_omega_n = omega_n * self.omega_max
        
        return output_v_n, output_omega_n