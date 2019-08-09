import torch
import torch.nn as nn

class PoseRefineNet(nn.Module) :
    def __init__(self, num_classes) :
        super(PoseRefineNet, self).__init__()
        ## input size 확인

        # 1*1 Convolution remaps the heatmaps to match the number of channels of the intermediate features
        self.pose_remap = nn.Sequential(
                nn.Conv2d(16, 128, 1, 1),
                nn.ReLU())
        self.parsing_remap = nn.Sequential(
                nn.Conv2d(num_classes, 128, 1, 1),
                nn.ReLU())

        # Get pose context 2
        self.pose_refine1 = nn.Sequential(
                nn.Conv2d(512, 512, 3, 1, padding=1),
                nn.ReLU())
        self.pose_refine2 = nn.Sequential(
                nn.Conv2d(512, 256, 5, 1, padding=2),
                nn.ReLU())
        self.pose_refine3 = nn.Sequential(
                nn.Conv2d(256, 256, 7, 1, padding=3),
                nn.ReLU())
        self.pose_refine4 = nn.Sequential(
                nn.Conv2d(256, 256, 9, 1, padding=4),
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
        pose_out = torch.cat((pose_out, parsing_out, pose_context), 1)

        pose_out = self.pose_refine1(pose_out)
        pose_out = self.pose_refine2(pose_out)
        pose_out = self.pose_refine3(pose_out)
        pose_context = self.pose_refine4(pose_out)
        pose_out = self.pose_refine5(pose_context)
        pose_out = self.pose_refine6(pose_out)

        return pose_out, pose_context

class ParsingRefineNet(nn.Module) :
    def __init__(self, num_classes) :
        super(ParsingRefineNet, self).__init__()

        # 1*1 Convolution remaps the heatmaps to match the number of channels of the intermediate features
        self.pose_remap = nn.Sequential(
                nn.Conv2d(16, 128, 1, 1),
                nn.ReLU())
        self.parsing_remap = nn.Sequential(
                nn.Conv2d(num_classes, 128, 1, 1),
                nn.ReLU())

        # Get parsing context 2
        self.parsing_refine1 = nn.Sequential(
                nn.Conv2d(512, 512, 3, 1, padding=1),
                nn.ReLU())
        self.parsing_refine2 = nn.Sequential(
                nn.Conv2d(512, 256, 5, 1, padding=2),
                nn.ReLU())
        self.parsing_refine3 = nn.Sequential(
                nn.Conv2d(256, 256, 7, 1, padding=3),
                nn.ReLU())
        self.parsing_refine4 = nn.Sequential(
                nn.Conv2d(256, 256, 9, 1, padding=4),
                nn.ReLU())

        # Get parsing map 2
        self.parsing_refine5 = nn.Sequential(
                nn.Conv2d(256, 256, 1, 1),
                nn.ReLU())
        self.parsing_atrous1 = nn.Conv2d(256, 20, 3, dilation=6, padding=6)
        self.parsing_atrous2 = nn.Conv2d(256, 20, 3, dilation=12, padding=12)
        self.parsing_atrous3 = nn.Conv2d(256, 20, 3, dilation=18, padding=18)
        self.parsing_atrous4 = nn.Conv2d(256, 20, 3, dilation=24, padding=24)

    def forward(self, pose_map, parsing_map, parsing_context) :
        pose_out = self.pose_remap(pose_map)
        parsing_out = self.parsing_remap(parsing_map)

        # Concat two remapped feature maps and parsing_context
        parsing_out = torch.cat((pose_out, parsing_out, parsing_context), 1)

        parsing_out = self.parsing_refine1(parsing_out)
        parsing_out = self.parsing_refine2(parsing_out)
        parsing_out = self.parsing_refine3(parsing_out)
        parsing_context = self.parsing_refine4(parsing_out)
        parsing_out = self.parsing_refine5(parsing_context)
        parsing_human1 = self.parsing_atrous1(parsing_out)
        parsing_human2 = self.parsing_atrous2(parsing_out)
        parsing_human3 = self.parsing_atrous3(parsing_out)
        parsing_human4 = self.parsing_atrous4(parsing_out)
        parsing_map = parsing_human1 + parsing_human2 + parsing_human3 + parsing_human4

        return parsing_map, parsing_context

class PoseNet(nn.Module) :
    def __init__(self) :
        super(PoseNet, self).__init__()

        ## Get pose context 1
        self.pose_conv1 = nn.Sequential(
                nn.Conv2d(1024, 512, 3, 1, padding=1),
                nn.ReLU())
        self.pose_conv2 = nn.Sequential(
                nn.Conv2d(512, 512, 3, 1, padding=1),
                nn.ReLU())
        self.pose_conv3 = nn.Sequential(
                nn.Conv2d(512, 256, 3, 1, padding=1),
                nn.ReLU())
        self.pose_conv4 = nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, padding=1),
                nn.ReLU())
        self.pose_conv5 = nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, padding=1),
                nn.ReLU())
        self.pose_conv6 = nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, padding=1),
                nn.ReLU())

        ## Get pose map 1
        self.pose_conv7 = nn.Sequential(
                nn.Conv2d(256, 512, 1, 1),
                nn.ReLU())
        self.pose_conv8 = nn.Sequential(
                nn.Conv2d(512, 16, 3, 1, padding=1))


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
    def __init__(self, num_classes) :
        super(ParsingNet, self).__init__()
        self.NUM_CLASS = num_classes
        self.ReLU = nn.ReLU()
        self.bn5a_branch1 = nn.Sequential(
            nn.Conv2d(1024, 2048, 1, stride=1, bias=False),
            nn.BatchNorm2d(2048),
        )
        self.bn5a_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 512, 1, 1, 512, 3, 4, 2048, 1, 1)
        )
        self.bn5b_branch2c = nn.Sequential(
            *self.atrousConvBn3(2048, 512, 1, 1, 512, 3, 4, 2048, 1, 1)
        )
        self.bn5c_branch2c = nn.Sequential(
            *self.atrousConvBn3(2048, 512, 1, 1, 512, 3, 4, 2048, 1, 1)
        )
        self.fc1_human_c0 = nn.Conv2d(2048, self.NUM_CLASS, 3, dilation=6, padding=6)
        self.fc1_human_c1 = nn.Conv2d(2048, self.NUM_CLASS, 3, dilation=12, padding=12)
        self.fc1_human_c2 = nn.Conv2d(2048, self.NUM_CLASS, 3, dilation=18, padding=18)
        self.fc1_human_c3 = nn.Conv2d(2048, self.NUM_CLASS, 3, dilation=24, padding=24)
        self.res5d_branch_parsing = nn.Sequential(
            nn.Conv2d(2048, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(),
        )

    def atrousConvBn3(self, channel_in, channel_1, kernel_1, stride_1,
                channel_2, kernel_2, rate,
                channel_3, kernel_3, stride_3) :
        return [
            nn.Conv2d(channel_in, channel_1, kernel_1, stride_1, bias=False),
            nn.BatchNorm2d(channel_1),
            nn.ReLU(),

            nn.Conv2d(channel_1, channel_2, kernel_2, dilation=rate, bias=False, padding=rate),
            nn.BatchNorm2d(channel_2),
            nn.ReLU(),

            nn.Conv2d(channel_2, channel_3, kernel_3, stride_3, bias=False),
            nn.BatchNorm2d(channel_3),
        ]

    def add(self, tensor1, tensor2) :
        return self.ReLU(torch.add(tensor1, tensor2))


    def forward(self, x) :
        branch_out = self.bn5a_branch1(x)
        operation_out = self.bn5a_branch2c(x)
        add_out = self.add(branch_out, operation_out)
        branch_out = self.bn5b_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn5c_branch2c(add_out)

        add_out = self.add(add_out, branch_out)

        aspp0 = self.fc1_human_c0(add_out)
        aspp1 = self.fc1_human_c1(add_out)
        aspp2 = self.fc1_human_c2(add_out)
        aspp3 = self.fc1_human_c3(add_out)

        fc1_human_out = aspp0 + aspp1 + aspp2 + aspp3
        res5d_branch_parsing_out = self.res5d_branch_parsing(add_out)

        return fc1_human_out, res5d_branch_parsing_out
