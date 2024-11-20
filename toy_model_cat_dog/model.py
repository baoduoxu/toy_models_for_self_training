import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        # 定义卷积层1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  # 输出32个特征图
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 定义池化层

        # 定义卷积层2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # 输出64个特征图

        # 定义全连接层
        self.fc1 = nn.Linear(64 * 32 * 32, 256)  # 假设输入图像大小为128x128
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 通过卷积层1和池化层
        x = self.pool(torch.relu(self.conv1(x)))
        # 通过卷积层2和池化层
        x = self.pool(torch.relu(self.conv2(x)))

        # 展平图像张量为一维向量
        x = x.view(x.size(0), -1)

        # 通过全连接层并激活
        x = torch.relu(self.fc1(x))
        # 通过输出层
        x = self.fc2(x)
        return x


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
