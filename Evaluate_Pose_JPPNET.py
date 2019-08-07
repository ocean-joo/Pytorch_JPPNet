import torch
import torch.nn as nn
import numpy as np
from LIP_Model import PoseRefineNet, ParsingRefineNet, PoseNet

img = Image.open(img_path)
transform = torch.Compose([
    torch.ToTensor()])
img = transform(img)


pose_net = PoseNet()
pose_refine = PoseRefineNet()
parsing_refine = ParsingRefineNet()

parsing_fea1, parsing_out1, resnet_fea1 = JPPNetModel([img])
pose_map1, pose_context1 = pose_net(resnet_fea1)
pose_map2, pose_context2 = pose_refine(pose_map1, parsing_out1, pose_context1)
parsing_out2, parsing_fea2 = parsing_refine(pose_map1, parsing_out1, parsing_fea1)
pose_map3, pose_context3 = pose_refine(pose_map2, parsing_out2, pose_context2)

