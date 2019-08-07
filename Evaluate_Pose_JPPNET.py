import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from LIP_Model import PoseRefineNet, ParsingRefineNet, PoseNet, ParsingNet
from ResNet_extract import BackboneResNet
from Model import JPPNetModel

parsing_net = ParsingNet(20)
pose_net = PoseNet()
pose_refine = PoseRefineNet(20)
parsing_refine = ParsingRefineNet(20)
model = JPPNetModel()
#model = BackboneResNet()

img = Image.open("2583_462340.jpg")
transform = transforms.Compose([transforms.ToTensor()])
img = transform(img)
img = img.unsqueeze(0)

feature_map = model(img)
parsing_out1, parsing_fea1 = parsing_net(feature_map)
#resnet_fea1, parsing_out1, parsing_fea1 = model(img)

pose_map1, pose_context1 = pose_net(feature_map)
pose_map2, pose_context2 = pose_refine(pose_map1, parsing_out1, pose_context1)
parsing_out2, parsing_fea2 = parsing_refine(pose_map1, parsing_out1, parsing_fea1)
pose_map3, pose_context3 = pose_refine(pose_map2, parsing_out2, pose_context2)
