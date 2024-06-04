import torch
from torch import nn


class DirectConnectModule(nn.Module):
    def __init__(self):
        super(DirectConnectModule, self).__init__()

    def forward(self, x):
        return x


class PreModel(nn.Module):
    def __init__(self):
        super(PreModel2, self).__init__()
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x
