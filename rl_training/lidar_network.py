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
        return x