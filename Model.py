import torch.nn as nn
from .ResNet_extract import *
from .utils import *

class PoseRefineNet(nn.Module) :
    def __init__(self, input_) :
        super(PoseRefineNet, self).__init__()
        ## input size 확인
        
        # 1*1 Convolution remaps the heatmaps to match the number of channels of the intermediate features
        self.pose_remap = nn.Sequential(
                nn.Conv2d(pose_size, 128, 1, 1),
                nn.ReLU())
        self.parsing_remap = nn.Sequential(
                nn.Conv2d(parsing_size, 128, 1, 1),
                nn.ReLU())

        # Concat two remapped feature maps


        # Get pose context 2
        self.pose_refine1 = nn.Sequential(
                nn.Conv2d(concat_size, 512, 3, 1),
                nn.ReLU())
        self.pose_refine2 = nn.Sequential(
                nn.Conv2d(512, 256, 5, 1),
                nn.ReLU())
        self.pose_refine3 = nn.Sequential(
                nn.Conv2d(256, 256, 7, 1),
                nn.ReLU())
        self.pose_refine4 = nn.Sequential(
                nn.Conv2d(256, 256, 9, 1),
                nn.ReLU())

        # Get pose map 2
        self.pose_refine5 = nn.Sequential(
                nn.Conv2d(256, 256, 1, 1),
                nn.ReLU())
        self.pose_refine6 = nn.Sequential(
                nn.Conv2d(256, 16, 1, 1))

    def forward(self, pose_map, parsing_map, pose_context) :
        pose_out = self.pose_remap(pose_map)
        parsing_out = self.parsing_remap(parsing_map)

        # Concat two remapped feature maps and pose_context
        pose_out = nn.

        return output, feature

class ParsingRefineNet(nn.Module) :
    def __init__(self) :
        super(ParsingRefineNet, self).__init__()

    def forward(self, x) :


class PoseNet(nn.Module) :
    def __init__(self) :
        super(PoseNet, self).__init__()
        
        ## Get pose context 1
        self.pose_conv1 = nn.Sequential(
                nn.Conv2d(input_size, 512, 3, 1),
                nn.ReLU())
        self.pose_conv2 = nn.Sequential(
                nn.Conv2d(512, 512, 3, 1),
                nn.ReLU())
        self.pose_conv3 = nn.Sequential(
                nn.Conv2d(512, 256, 3, 1),
                nn.ReLU())
        self.pose_conv4 = nn.Sequential(
                nn.Conv2d(256, 256, 3, 1),
                nn.ReLU())
        self.pose_conv5 = nn.Sequential(
                nn.Conv2d(256, 256, 3, 1),
                nn.ReLU())
        self.pose_conv6 = nn.Sequential(
                nn.Conv2d(256, 256, 3, 1),
                nn.ReLU())

        ## Get pose map 1
        self.pose_conv7 = nn.Sequential(
                nn.Conv2d(256, 512, 1, 1),
                nn.ReLU())
        self.pose_conv8 = nn.Sequential(
                nn.Conv2d(512, 16, 3, 1))


    def forward(self, x) :
        pose_out = self.pose_conv1(x)
        pose_out = self.pose_conv2(pose_out)
        pose_out = self.pose_conv3(pose_out)
        pose_out = self.pose_conv4(pose_out)
        pose_out = self.pose_conv5(pose_out)
        pose_context = self.pose_conv6(pose_out)
        pose_out = self.pose_conv7(pose_context)
        pose_out = self.pose_conv8(pose_out)

        return pose_out, pose_context

class ParsingNet(nn.Module) :
    def __init__(self) :
        super(ParsingNet, self).__init()

    def forward(self, x) :
        pass
