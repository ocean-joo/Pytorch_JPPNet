import torch
import torch.nn as nn
import torch.optim as optim

class BackboneResNet(nn.Module) :
    def __init__(self) :
        super(BackboneResNet, self).__init__()
        module = self.convBn(3, 64, 7, 2).extend(nn.MaxPool2d(3, stride=2))
        module.extend(self.convBn(64, 256, 1, 1, activation=False))
        self.block1 = nn.Sequential(*module)
        self.block2 = nn.Sequential(*self.convBn3(256, 64, 1, 1, 64, 3, 1, 256, 1, 1))
        module = [nn.ReLU()].extend(self.convBn3(256, 64, 1, 1, 64, 3, 1, 256, 1, 1))
        self.block3 = nn.Sequential(*module)
        self.block4 =

    def convBn3(channel_in, channel_1, kernel_1, stride_1,
                channel_2, kernel_2, stride_2,
                channel_3, kernel_3, stride_3) :
        return [
            nn.Conv2d(channel_in, channel_1, kernel_1, stride_2, bias=False),
            nn.BatchNorm2d(channel_1),
            nn.ReLU(),

            nn.Conv2d(channel_1, channel_2, kernel_2, stride_2, bias=False),
            nn.BatchNorm2d(channel_2),
            nn.ReLU(),

            nn.Conv2d(channel_2, channel_3, kernel_3, stride_3, bias=False),
            nn.BatchNorm2d(channel_3),
        ]

    def convBn(channel_in, kernel, channel_out, stride, activation=True) :
        module = [
            nn.Conv2d(channel_in, channel_out, kernel, stride, bias=False),
            nn.BatchNorm2d(channel_out),
        ]
        if activation is True :
            module.append(nn.ReLu())
        return module

    def forward(self, x) :
        x =
