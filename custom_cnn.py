# custom_cnn.py
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64): # features_dim 可以调整
        # features_dim 是 CNN 提取特征后的输出维度，将连接到后续的策略和价值网络
        super().__init__(observation_space, features_dim)

        # 我们假设输入 observation_space.shape == (1, H, W)
        n_input_channels = observation_space.shape[0] # 应该是 1
        height = observation_space.shape[1]
        width = observation_space.shape[2]

        # 定义 CNN 层
        # 这是一个示例结构，你可以根据需要调整卷积核大小、步长、填充、层数、激活函数等
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1), # 输出: (32, H, W)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),              # 输出: (64, H, W)
            nn.ReLU(),
            # 可以添加更多卷积层或池化层 (nn.MaxPool2d)
            # nn.MaxPool2d(kernel_size=2, stride=2), # 例如，如果添加池化
            nn.Flatten(), # 将特征图展平成一维向量
        )

        # 计算展平后的特征维度
        # 需要手动运行一次前向传播来获取维度，或者根据 H, W 和卷积/池化层参数计算
        # 假设输入尺寸是 1x10x10 (示例)
        with torch.no_grad():
            # 创建一个假的输入张量来计算输出形状
            # 注意：这里的 batch_size=1 是为了计算，实际训练时 batch_size 会不同
            dummy_input = torch.zeros(1, n_input_channels, height, width)
            n_flatten = self.cnn(dummy_input).shape[1] # 获取展平后的维度

        # 定义连接到最终输出维度的线性层
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU() # 通常在特征提取后加一个激活函数
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param observations: (Tensor) 输入观测值，shape=(batch_size, C, H, W)
        :return: (Tensor) 提取的特征，shape=(batch_size, features_dim)
        """
        # 注意：observations 已经是 float32 类型，并且应该已经被归一化了 (在环境中完成)
        cnn_output = self.cnn(observations)
        features = self.linear(cnn_output)
        return features