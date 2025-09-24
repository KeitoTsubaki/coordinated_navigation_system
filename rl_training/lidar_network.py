import torch
from torch import nn
import math


class LiDARNetwork(nn.Module):
    def __init__(self):
        super(LiDARNetwork, self).__init__()
        # 静的データ用 (1チャンネル入力)
        self.static_conv1 = nn.Conv1d(
            in_channels=1, out_channels=8, kernel_size=5, stride=2
        )
        self.static_conv2 = nn.Conv1d(
            in_channels=8, out_channels=16, kernel_size=5, stride=2
        )
        self.static_conv3 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, stride=2
        )

        # 動的データ用 (4チャンネル入力)
        self.dynamic_conv1 = nn.Conv1d(
            in_channels=4, out_channels=8, kernel_size=5, stride=2
        )
        self.dynamic_conv2 = nn.Conv1d(
            in_channels=8, out_channels=16, kernel_size=5, stride=2
        )
        self.dynamic_conv3 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, stride=2
        )

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.leaky_relu = nn.LeakyReLU(0.01)

        # Calculate the flattened size dynamically
        with torch.no_grad():
            cnn_output_dim = self._get_conv_output_dim(
                torch.zeros(1, 1, 640)
            )  # 仮の入力サイズ
        self.static_fc = nn.Linear(cnn_output_dim, 40)
        self.dynamic_fc = nn.Linear(cnn_output_dim, 20)

    def _get_conv_output_dim(self, x):
        # Helper function to calculate the output dimension of the CNN layers
        x = self.static_conv1(x)
        x = self.static_conv2(x)
        x = self.static_conv3(x)
        x = self.maxpool(x)
        return x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        # e.g., (32, 4, 640)
        static_x = x[
            :, -1:, :
        ]  # Last channel for static data (shape: (batch_size, 1, 640))

        static_output = self._forward_static(static_x)
        dynamic_output = self._forward_dynamic(x)

        lidar_feature = torch.cat(
            (static_output, dynamic_output), dim=1
        )  # (shape: (1, 60))

        return lidar_feature

    def _forward_static(self, static_x):
        # static_x shape: (batch_size, 1, sequence_length)
        static_x = self.leaky_relu(self.static_conv1(static_x))
        static_x = self.leaky_relu(self.static_conv2(static_x))
        static_x = self.leaky_relu(self.static_conv3(static_x))
        static_x = self.maxpool(static_x)
        static_x_flatten = static_x.view(
            static_x.size(0), -1
        )  # Flatten (shape: (1, 1216))
        static_output = self.leaky_relu(self.static_fc(static_x_flatten))
        return static_output

    def _forward_dynamic(self, dynamic_x):
        # dynamic_x shape: (batch_size, 4, sequence_length)
        dynamic_x = self.leaky_relu(self.dynamic_conv1(dynamic_x))
        dynamic_x = self.leaky_relu(self.dynamic_conv2(dynamic_x))
        dynamic_x = self.leaky_relu(self.dynamic_conv3(dynamic_x))
        dynamic_x = self.maxpool(dynamic_x)
        dynamic_x_flatten = dynamic_x.view(
            dynamic_x.size(0), -1
        )  # Flatten (shape: (1, 1216))
        dynamic_output = self.leaky_relu(self.dynamic_fc(dynamic_x_flatten))
        return dynamic_output
