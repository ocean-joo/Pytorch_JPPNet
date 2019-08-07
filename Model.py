import torch
import torch.nn as nn

class JPPNetModel(nn.Module) :
    def __init__(self) :
        super(JPPNetModel, self).__init__()

        self.ReLU = nn.ReLU()

        self.layer1_1 = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2))
        self.layer1_2 = nn.Sequential(
                nn.Conv2d(64, 256, 1, 1),
                nn.BatchNorm2d(256))

        self.layer2 = nn.Sequential(
                nn.Conv2d(64, 64, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 256, 1, 1),
                nn.BatchNorm2d(256))

        self.layer3 = nn.Sequential(
                nn.Conv2d(256, 64, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 256, 1, 1),
                nn.BatchNorm2d(256))

        self.layer4 = nn.Sequential(
                nn.Conv2d(256, 512, 1, 2),
                nn.BatchNorm2d(512))

        self.layer5 = nn.Sequential(
                nn.Conv2d(256, 128, 1, 2),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, 1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 512, 1, 1),
                nn.BatchNorm2d(512))
        
        self.layer6 = nn.Sequential(
                nn.Conv2d(512, 128, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, 1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 512, 1, 1),
                nn.BatchNorm2d(512))

        self.layer7 = nn.Sequential(
                nn.Conv2d(512, 1024, 1, 1),
                nn.BatchNorm2d(1024))

        self.layer8 = nn.Sequential(
                nn.Conv2d(512, 256, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, dilation=2, padding=2),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 1024, 1, 1),
                nn.BatchNorm2d(1024))

        self.layer9 = nn.Sequential(
                nn.Conv2d(1024, 256, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, dilation=2, padding=2),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 1024, 1, 1),
                nn.BatchNorm2d(1024))

    def forward(self, x) :
        resnet_out1 = self.layer1_1(x)
        resnet_out2 = self.layer1_2(resnet_out1)

        resnet_out1 = self.layer2(resnet_out1)


        for i in range(2):
            resnet_out = torch.add(resnet_out1, resnet_out2)
            resnet_out1 = self.ReLU(resnet_out)
            resnet_out2 = self.layer3(resnet_out1)

        resnet_out = resnet_out1 + resnet_out2
        resnet_out = self.ReLU(resnet_out)
        resnet_out1 = self.layer4(resnet_out)
        resnet_out2 = self.layer5(resnet_out)

        for i in range(3):
            resnet_out = resnet_out1 + resnet_out2
            resnet_out1 = self.ReLU(resnet_out)
            resnet_out2 = self.layer6(resnet_out1)
        
        resnet_out = resnet_out1 + resnet_out2
        resnet_out = self.ReLU(resnet_out)
        resnet_out1 = self.layer7(resnet_out)
        resnet_out2 = self.layer8(resnet_out)

        for i in range(22):
            resnet_out = resnet_out1 + resnet_out2
            resnet_out1 = self.ReLU(resnet_out)
            resnet_out2 = self.layer9(resnet_out1)

        resnet_out = resnet_out1 + resnet_out2
        resnet_out = self.ReLU(resnet_out)

        return resnet_out












