import torch
import torch.nn as nn

class BackboneResNet(nn.Module) :
    def __init__(self) :
        super(BackboneResNet, self).__init__()
        self.NUM_CLASS = 20
        self.ReLU = nn.ReLU()
        self.pool1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, stride=2),
        )
        self.bn2a_branch1 = nn.Sequential(
            nn.Conv2d(64, 256, 1, stride=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.bn2a_branch2c = nn.Sequential(
            *self.convBn3(64, 64, 1, 1, 64, 3, 1, 256, 1, 1)
        )
        self.bn2b_branch2c = nn.Sequential(
            *self.convBn3(256, 64, 1, 1, 64, 3, 1, 256, 1, 1)
        )
        self.bn2c_branch2c = nn.Sequential(
            *self.convBn3(256, 64, 1, 1, 64, 3, 1, 256, 1, 1)
        )
        self.bn3a_branch1 = nn.Sequential(
            nn.Conv2d(256, 512, 1, stride=2, bias=False),
            nn.BatchNorm2d(512),
        )
        self.bn3a_branch2c = nn.Sequential(
            *self.convBn3(256, 128, 1, 2, 128, 3, 1, 512, 1, 1)
        )
        self.bn3b1_branch2c = nn.Sequential(
            *self.convBn3(512, 128, 1, 1, 128, 3, 1, 512, 1, 1)
        )
        self.bn3b2_branch2c = nn.Sequential(
            *self.convBn3(512, 128, 1, 1, 128, 3, 1, 512, 1, 1)
        )
        self.bn3b3_branch2c = nn.Sequential(
            *self.convBn3(512, 128, 1, 1, 128, 3, 1, 512, 1, 1)
        )
        self.bn4a_branch1 = nn.Sequential(
            nn.Conv2d(512, 1024, 1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.bn4a_branch2c = nn.Sequential(
            *self.atrousConvBn3(512, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b1_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b2_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b3_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b4_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b5_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b6_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b7_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b8_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b9_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b10_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b11_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b12_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b13_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b14_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b15_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b16_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b17_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b18_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b19_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b20_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b21_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
        self.bn4b22_branch2c = nn.Sequential(
            *self.atrousConvBn3(1024, 256, 1, 1, 256, 3, 2, 1024, 1, 1)
        )
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
            nn.Conv2d(2048, 512, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, stride=1),
            nn.ReLU(),
        )

    def convBn3(self, channel_in, channel_1, kernel_1, stride_1,
                channel_2, kernel_2, stride_2,
                channel_3, kernel_3, stride_3) :
        return [
            nn.Conv2d(channel_in, channel_1, kernel_1, stride_1, bias=False),
            nn.BatchNorm2d(channel_1),
            nn.ReLU(),

            nn.Conv2d(channel_1, channel_2, kernel_2, stride_2, bias=False, padding=1),
            nn.BatchNorm2d(channel_2),
            nn.ReLU(),

            nn.Conv2d(channel_2, channel_3, kernel_3, stride_3, bias=False),
            nn.BatchNorm2d(channel_3),
        ]

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
        operation_out = self.pool1(x)
        branch_out = self.bn2a_branch1(operation_out)
        operation_out = self.bn2a_branch2c(operation_out)

        print(branch_out.size())
        print(operation_out.size())
        add_out = self.add(branch_out, operation_out)
        branch_out = self.bn2b_branch2c(add_out)

        add_out = self.add(branch_out, add_out)
        branch_out = self.bn2c_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn3a_branch1(add_out)
        add_out = self.bn3a_branch2c(add_out)

        add_out = self.add(branch_out, add_out)
        branch_out = self.bn3b1_branch2c(add_out)

        add_out = self.add(branch_out, add_out)
        branch_out = self.bn3b2_branch2c(add_out)

        add_out = self.add(branch_out, add_out)
        branch_out = self.bn3b3_branch2c(add_out)

        add_out = self.add(branch_out, add_out)
        branch_out = self.bn4a_branch1(add_out)

        operation_out = self.bn4a_branch2c(add_out)
        add_out = self.add(branch_out, operation_out)
        branch_out = self.bn4b1_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b2_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b3_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b4_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b5_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b6_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b7_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b8_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b9_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b10_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b11_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b12_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b13_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b14_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b15_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b16_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b17_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b18_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b19_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b20_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b21_branch2c(add_out)

        add_out = self.add(add_out, branch_out)
        branch_out = self.bn4b22_branch2c(add_out)

        res4b22_relu_out = self.add(add_out, branch_out)

        branch_out = self.bn5a_branch1(res4b22_relu_out)
        operation_out = self.bn5a_branch2c(res4b22_relu_out)
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

        return res4b22_relu_out, fc1_human_out, res5d_branch_parsing_out
