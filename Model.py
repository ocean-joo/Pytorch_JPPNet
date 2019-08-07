import torch.nn as nn
from .ResNet_extract import *
from .utils import *

class PoseRefineNet(nn.Module) :
    def __init__(self, input_) :
        super(PoseRefineNet, self).__init__()
        ## input size 확인
        self.pose_conv1 = nn.Conv2d(input_, 512, 3, stride=1)
        self.pose_conv2 = nn.Conv2d(512, 512, 3, stride=1)
        self.pose_conv3 = nn.Conv2d(512, 256, 3, stride=1)
        self.pose_conv4 = nn.Conv2d(512, 256, 3, stride=1)
        self.pose_conv5 = nn.Conv2d()
        self.pose_conv6 = nn.Conv2d()
        self.pose_conv7 = nn.Conv2d()
        self.pose_conv8 = nn.Conv2d()

    def forward(self, x) :

        return output, feature

class ParsingRefineNet(nn.Module) :
    def __init__(self) :
        super(ParsingRefineNet, self).__init__()

    def forward(self, x) :


class PoseNet(nn.Module) :
    def __init__(self) :
        super(PoseNet, self).__init__()

    def forward(self, x) :

class ParsingNet(nn.Module) :
    def __init__(self) :
        super(ParsingNet, self).__init()

    def forward(self, x) :
        pass
