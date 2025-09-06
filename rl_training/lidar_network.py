import torch
from torch import nn

class LiDARNetwork(nn.Module):
    def __init__(self):
        super(LiDARNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.leaky_relu = nn.LeakyReLU(0.01)
        
        self.static_fc = nn.Linear(1216, 40)
        self.dynamic_fc = nn.Linear(1216, 20)

    def forward(self, x):
        # x shape: (1, 4, 460) -> (batch_size, channels, sequence_length)
        static_x = x[:, -1:, :]  # Last channel for static data (shape: (1, 1, 460))
        
        static_output = self.static_net(static_x)  # (shape: (1, 40))
        dynamic_output = self.dynamic_net(x)
        
        lidar_feature = torch.cat((static_output, dynamic_output), dim=1)  # (shape: (1, 60))
        
        return lidar_feature
    
    def static_net(self, static_x):
        # static_x shape: (1, 1, 460)
        static_x = self.leaky_relu(self.conv1(static_x))
        static_x = self.leaky_relu(self.conv2(static_x))
        static_x = self.leaky_relu(self.conv3(static_x))
        static_x = self.maxpool(static_x)
        static_x_flatten = static_x.view(static_x.size(0), -1)  # Flatten (shape: (1, 1216))
        static_output = self.leaky_relu(self.static_fc(static_x_flatten)) 
        return static_output
    
    def dynamic_net(self, dynamic_x):
        # dynamic_x shape: (1, 4, 460)
        dynamic_x = self.leaky_relu(self.conv1(dynamic_x))
        dynamic_x = self.leaky_relu(self.conv2(dynamic_x))
        dynamic_x = self.leaky_relu(self.conv3(dynamic_x))
        dynamic_x = self.maxpool(dynamic_x)
        dynamic_x_flatten = dynamic_x.view(dynamic_x.size(0), -1)  # Flatten (shape: (1, 1216))
        dynamic_output = self.leaky_relu(self.dynamic_fc(dynamic_x_flatten)) 
        return dynamic_output