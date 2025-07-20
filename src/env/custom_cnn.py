# custom_cnn.py
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    """
    Adaptive CNN that can handle variable input sizes for minesweeper boards.
    Uses adaptive pooling to produce consistent output regardless of input dimensions.
    
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        # features_dim 是 CNN 提取特征后的输出维度，将连接到后续的策略和价值网络
        super().__init__(observation_space, features_dim)

        # 我们假设输入 observation_space.shape == (1, H, W)
        n_input_channels = observation_space.shape[0] # 应该是 1

        # 定义 CNN 层 - 使用卷积层提取特征，不依赖于输入尺寸
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1), # 输出: (32, H, W)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),              # 输出: (64, H, W)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),             # 输出: (128, H, W)
            nn.ReLU(),
        )
        
        # 使用自适应池化层来处理任意输入尺寸
        # 这将把任何 (128, H, W) 的特征图转换为 (128, 4, 4) 的固定尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 计算池化后的特征维度: 128 channels * 4 * 4 = 2048
        pooled_features_dim = 128 * 4 * 4
        
        # 定义连接到最终输出维度的线性层
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pooled_features_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加dropout防止过拟合
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 现在支持任意输入尺寸
        :param observations: (Tensor) 输入观测值，shape=(batch_size, C, H, W)
        :return: (Tensor) 提取的特征，shape=(batch_size, features_dim)
        """
        # 通过卷积层提取特征
        cnn_output = self.cnn_layers(observations)
        
        # 使用自适应池化统一特征图尺寸
        pooled_output = self.adaptive_pool(cnn_output)
        
        # 通过线性层得到最终特征
        features = self.linear(pooled_output)
        return features