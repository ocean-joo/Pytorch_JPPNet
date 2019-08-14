import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os
import time
import numpy as np
from PIL import Image

from LIP_Model import PoseRefineNet, ParsingRefineNet, PoseNet, ParsingNet
from Model import BackboneResNet
from dataloader import LIPDataset, myTensor

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
device = "cuda" if torch.cuda.is_available else "cpu"
#device = "cpu"

N_CLASSES = 20
INPUT_SIZE = (384, 384)
BATCH_SIZE = 3
SHUFFLE = True
RANDOM_SCALE = True
RANDOM_MIRROR = True
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
POWER = 0.9
NUM_STEPS = 7616 * 35 + 1
SAVE_PRED_EVERY = 7616
pose_weight = 1
parsing_weight = 1
EPOCH = 100

BASE_DIR = "/mnt/ssd/Data/LIP/"

parsing_net = ParsingNet(20).to(device)
pose_net = PoseNet().to(device)
pose_refine = PoseRefineNet(20).to(device)
parsing_refine = ParsingRefineNet(20).to(device)
backbone = BackboneResNet().to(device)

print("[*] Loading Model Params...")
backbone.load_state_dict(torch.load("parameters/backbone_parameter.pt"))
parsing_net.load_state_dict(torch.load("parameters/parsing_net_parameter.pt"))
parsing_refine.load_state_dict(torch.load("parameters/parsing_refine_net_parameter.pt"))
pose_net.load_state_dict(torch.load("parameters/pose_net_parameter.pt"))
pose_refine.load_state_dict(torch.load("parameters/pose_refine_net_parameter.pt"))
print("[!] Sucess")

base_lr = LEARNING_RATE
#learning_rate = tf.scalar_mul(base_lr, ((1 - step_ph / NUM_STEPS), POWER)**2)
params = list(parsing_net.parameters()) + list(pose_net.parameters()) + list(pose_refine.parameters()) + list(parsing_refine.parameters()) + list(backbone.parameters())

optimizer = torch.optim.RMSprop(params, lr=base_lr, momentum=MOMENTUM)

parsing_net = nn.DataParallel(parsing_net).train()
pose_net = nn.DataParallel(pose_net).train()
pose_refine = nn.DataParallel(pose_refine).train()
parsing_refine = nn.DataParallel(parsing_refine).train()
backbone = nn.DataParallel(backbone).train()

trans_resize = transforms.Compose([transforms.Resize(INPUT_SIZE)])
dset = LIPDataset(BASE_DIR, trans_resize)

data_loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

for e in range(EPOCH) :
    print("[!] Training EPOCH {}".format(e))
    for i, (orig_img, segment, heatmap) in enumerate(data_loader) :
        if isinstance(orig_img, torch.Tensor) :
            orig_img = orig_img.numpy()
            print(orig_img.shape)

        orig_img = Image.fromarray(orig_img)
        segment = segment.to(device)
        heatmap = heatmap.to(device)

        optimizer.zero_grad()

        transform = transforms.Compose([myTensor()])
        img_100 = transform(orig_img)
        orig_img_size = tuple(img_100.shape[1:])

        transform_075 = transforms.Compose([transforms.Resize(tuple([int(t*0.75) for t in orig_img_size])),
                                            myTensor()])
        img_075 = transform_075(orig_img)

        transform_050 = transforms.Compose([transforms.Resize(tuple([int(t*0.5) for t in orig_img_size])),
                                            myTensor()])
        img_050 = transform_050(orig_img)


        img_100 = img_100.unsqueeze(0)
        img_100_reverse = torch.flip(img_100, dims=[2])

        img_075 = img_075.unsqueeze(0)
        img_075_reverse = torch.flip(img_075, dims=[2])

        img_050 = img_050.unsqueeze(0)
        img_050_reverse = torch.flip(img_050, dims=[2])

        input_img_100 = img_100.to(device) # only use orig img
        input_img_075 = img_075.to(device)
        input_img_050 = img_050.to(device)

        feature_map_100 = backbone(input_img_100)
        parsing_map1_100, parsing_context1_100 = parsing_net(feature_map_100)
        pose_map1_100, pose_context1_100 = pose_net(feature_map_100)

        pose_map2_100, pose_context2_100 = pose_refine(pose_map1_100, parsing_map1_100, pose_context1_100)
        parsing_map2_100, parsing_context2_100 = parsing_refine(pose_map1_100, parsing_map1_100, parsing_context1_100)
        parsing_map3_100, parsing_context3_100 = parsing_refine(pose_map2_100, parsing_map2_100, parsing_context2_100)
        pose_map3_100, pose_context3_100 = pose_refine(pose_map2_100, parsing_map2_100, pose_context2_100)

        feature_map_075 = backbone(input_img_075)
        parsing_map1_075, parsing_context1_075 = parsing_net(feature_map_075)
        pose_map1_075, pose_context1_075 = pose_net(feature_map_075)

        pose_map2_075, pose_context2_075 = pose_refine(pose_map1_075, parsing_map1_075, pose_context1_075)
        parsing_map2_075, parsing_context2_075 = parsing_refine(pose_map1_075, parsing_map1_075, parsing_context1_075)
        parsing_map3_075, parsing_context3_075 = parsing_refine(pose_map2_075, parsing_map2_075, parsing_context2_075)
        pose_map3_075, pose_context3_075 = pose_refine(pose_map2_075, parsing_map2_075, pose_context2_075)

        feature_map_050 = backbone(input_img_050)
        parsing_map1_050, parsing_context1_050 = parsing_net(feature_map_050)
        pose_map1_050, pose_context1_050 = pose_net(feature_map_050)

        pose_map2_050, pose_context2_050 = pose_refine(pose_map1_050, parsing_map1_050, pose_context1_050)
        parsing_map2_050, parsing_context2_050 = parsing_refine(pose_map1_050, parsing_map1_050, parsing_context1_050)
        parsing_map3_050, parsing_context3_050 = parsing_refine(pose_map2_050, parsing_map2_050, parsing_context2_050)
        pose_map3_050, pose_context3_050 = pose_refine(pose_map2_050, parsing_map2_050, pose_context2_050)


        parsing_map1_size = parsing_map1_100.shape[2:4]
        parsing_map1 = torch.mean(torch.stack((parsing_map1_100,
                                             F.interpolate(parsing_map1_075, size=parsing_map1_size, mode="bilinear", align_corners=True),
                                             F.interpolate(parsing_map1_050, size=parsing_map1_size, mode="bilinear", align_corners=True))), dim=0)

        parsing_map2_size = parsing_map2_100.shape[2:4]
        parsing_map2 = torch.mean(torch.stack((parsing_map2_100,
                                             F.interpolate(parsing_map2_075, size=parsing_map2_size, mode="bilinear", align_corners=True),
                                             F.interpolate(parsing_map2_050, size=parsing_map2_size, mode="bilinear", align_corners=True))), dim=0)

        parsing_map3_size = parsing_map3_100.shape[2:4]
        parsing_map3 = torch.mean(torch.stack((parsing_map3_100,
                                             F.interpolate(parsing_map3_075, size=parsing_map3_size, mode="bilinear", align_corners=True),
                                             F.interpolate(parsing_map3_050, size=parsing_map3_size, mode="bilinear", align_corners=True))), dim=0)


        pose_map1_size = pose_map1_100.shape[2:4]
        pose_map1 = torch.mean(torch.stack((pose_map1_100,
                                             F.interpolate(pose_map1_100, size=pose_map1_size),
                                             F.interpolate(pose_map1_100, size=pose_map1_size))), dim=0)

        pose_map2_size = pose_map2_100.shape[2:4]
        pose_map2 = torch.mean(torch.stack((pose_map2_100,
                                             F.interpolate(pose_map2_100, size=pose_map2_size),
                                             F.interpolate(pose_map2_100, size=pose_map2_size))), dim=0)

        pose_map3_size = pose_map3_100.shape[2:4]
        pose_map3 = torch.mean(torch.stack((pose_map3_100,
                                             F.interpolate(pose_map3_100, size=pose_map3_size),
                                             F.interpolate(pose_map3_100, size=pose_map3_size))), dim=0)


        raw_prediction_p1 = parsing_map1.reshape(N_CLASSES, -1)
        raw_prediction_p1_100 = parsing_map1_100.reshape(N_CLASSES, -1)
        raw_prediction_p1_075 = parsing_map1_075.reshape(N_CLASSES, -1)
        raw_prediction_p1_050 = parsing_map1_050.reshape(N_CLASSES, -1)

        raw_prediction_p2 = parsing_map2.reshape(N_CLASSES, -1)
        raw_prediction_p2_100 = parsing_map2_100.reshape(N_CLASSES, -1)
        raw_prediction_p2_075 = parsing_map2_075.reshape(N_CLASSES, -1)
        raw_prediction_p2_050 = parsing_map2_050.reshape(N_CLASSES, -1)


        raw_prediction_p3 = parsing_map3.reshape(N_CLASSES, -1)
        raw_prediction_p3_100 = parsing_map3_100.reshape(N_CLASSES, -1)
        raw_prediction_p3_075 = parsing_map3_075.reshape(N_CLASSES, -1)
        raw_prediction_p3_050 = parsing_map3_050.reshape(N_CLASSES, -1)


        new_segment = segment.reshape(-1,1,segment.shape[1],segment.shape[2]).float()
        # print("new_segment", new_segment.shape)
        label_proc_100 = F.interpolate(new_segment, size=parsing_map1_100.shape[2:4]).int().squeeze()
        label_proc_075 = F.interpolate(new_segment, size=parsing_map1_075.shape[2:4]).int().squeeze()
        label_proc_050 = F.interpolate(new_segment, size=parsing_map1_050.shape[2:4]).int().squeeze()

        raw_gt_100 = label_proc_100.reshape(-1)
        raw_gt_075 = label_proc_075.reshape(-1)
        raw_gt_050 = label_proc_050.reshape(-1)

        # print("raw_gt shape ", raw_gt_100.shape)
        indices_100 = (raw_gt_100 <= N_CLASSES-1).nonzero().reshape(-1)
        indices_075 = (raw_gt_075 <= N_CLASSES-1).nonzero().reshape(-1)
        indices_050 = (raw_gt_050 <= N_CLASSES-1).nonzero().reshape(-1)


        gt_100 = torch.index_select(raw_gt_100, dim=0, index=indices_100)
        gt_075 = torch.index_select(raw_gt_075, dim=0, index=indices_075)
        gt_050 = torch.index_select(raw_gt_050, dim=0, index=indices_050)

        # print("raw ", raw_prediction_p1.shape)
        # print("indices ", indices_100)
        prediction_p1 = torch.index_select(raw_prediction_p1, dim=1, index=indices_100)
        prediction_p1_100 = torch.index_select(raw_prediction_p1_100, dim=1, index=indices_100)
        prediction_p1_075 = torch.index_select(raw_prediction_p1_075, dim=1, index=indices_075)
        prediction_p1_050 = torch.index_select(raw_prediction_p1_050, dim=1, index=indices_050)

        prediction_p2 = torch.index_select(raw_prediction_p2, dim=1, index=indices_100)
        prediction_p2_100 = torch.index_select(raw_prediction_p2_100, dim=1, index=indices_100)
        prediction_p2_075 = torch.index_select(raw_prediction_p2_075, dim=1, index=indices_075)
        prediction_p2_050 = torch.index_select(raw_prediction_p2_050, dim=1, index=indices_050)

        prediction_p3 = torch.index_select(raw_prediction_p3, dim=1, index=indices_100)
        prediction_p3_100 = torch.index_select(raw_prediction_p3_100, dim=1, index=indices_100)
        prediction_p3_075 = torch.index_select(raw_prediction_p3_075, dim=1, index=indices_075)
        prediction_p3_050 = torch.index_select(raw_prediction_p3_050, dim=1, index=indices_050)

        heatmap_100 = F.interpolate(heatmap, size=pose_map1_100.shape[2:4])
        heatmap_075 = F.interpolate(heatmap, size=pose_map1_075.shape[2:4])
        heatmap_050 = F.interpolate(heatmap, size=pose_map1_050.shape[2:4])


        loss_p1 = F.nll_loss(F.softmax(prediction_p1.unsqueeze(0), dim=1), gt_100.unsqueeze(0).long())
        loss_p1_100 = F.nll_loss(F.softmax(prediction_p1_100.unsqueeze(0), dim=1), gt_100.unsqueeze(0).long())
        loss_p1_075 = F.nll_loss(F.softmax(prediction_p1_075.unsqueeze(0), dim=1), gt_075.unsqueeze(0).long())
        loss_p1_050 = F.nll_loss(F.softmax(prediction_p1_050.unsqueeze(0), dim=1), gt_050.unsqueeze(0).long())

        loss_p2 = F.nll_loss(F.softmax(prediction_p2.unsqueeze(0), dim=1), gt_100.unsqueeze(0).long())
        loss_p2_100 = F.nll_loss(F.softmax(prediction_p2_100.unsqueeze(0), dim=1), gt_100.unsqueeze(0).long())
        loss_p2_075 = F.nll_loss(F.softmax(prediction_p2_075.unsqueeze(0), dim=1), gt_075.unsqueeze(0).long())
        loss_p2_050 = F.nll_loss(F.softmax(prediction_p2_050.unsqueeze(0), dim=1), gt_050.unsqueeze(0).long())

        loss_p3 = F.nll_loss(F.softmax(prediction_p3.unsqueeze(0), dim=1), gt_100.unsqueeze(0).long())
        loss_p3_100 = F.nll_loss(F.softmax(prediction_p3_100.unsqueeze(0), dim=1), gt_100.unsqueeze(0).long())
        loss_p3_075 = F.nll_loss(F.softmax(prediction_p3_075.unsqueeze(0), dim=1), gt_075.unsqueeze(0).long())
        loss_p3_050 = F.nll_loss(F.softmax(prediction_p3_050.unsqueeze(0), dim=1), gt_050.unsqueeze(0).long())


        loss_s1 = torch.mean(torch.sqrt(torch.sum(((heatmap_100 - pose_map1)**2), (1,2,3))))
        loss_s1_100 = torch.mean(torch.sqrt(torch.sum(((heatmap_100 - pose_map1_100)**2), (1,2,3))))
        loss_s1_075 = torch.mean(torch.sqrt(torch.sum(((heatmap_075 - pose_map1_075)**2), (1,2,3))))
        loss_s1_050 = torch.mean(torch.sqrt(torch.sum(((heatmap_050 - pose_map1_050)**2), (1,2,3))))


        loss_s2 = torch.mean(torch.sqrt(torch.sum(((heatmap_100 - pose_map2)**2), (1,2,3))))
        loss_s2_100 = torch.mean(torch.sqrt(torch.sum(((heatmap_100 - pose_map2_100)**2), (1,2,3))))
        loss_s2_075 = torch.mean(torch.sqrt(torch.sum(((heatmap_075 - pose_map2_075)**2), (1,2,3))))
        loss_s2_050 = torch.mean(torch.sqrt(torch.sum(((heatmap_050 - pose_map2_050)**2), (1,2,3))))


        loss_s3 = torch.mean(torch.sqrt(torch.sum(((heatmap_100 - pose_map3)**2), (1,2,3))))
        loss_s3_100 = torch.mean(torch.sqrt(torch.sum(((heatmap_100 - pose_map3_100)**2), (1,2,3))))
        loss_s3_075 = torch.mean(torch.sqrt(torch.sum(((heatmap_075 - pose_map3_075)**2), (1,2,3))))
        loss_s3_050 = torch.mean(torch.sqrt(torch.sum(((heatmap_050 - pose_map3_050)**2), (1,2,3))))


        loss_parsing = loss_p1 + loss_p1_100 + loss_p1_075 + loss_p1_050 + loss_p2 + loss_p2_100 + loss_p2_075 + loss_p2_050 + loss_p3 + loss_p3_100 + loss_p3_075 + loss_p3_050
        loss_pose = loss_s1 + loss_s1_100 + loss_s1_075 + loss_s1_050 + loss_s2 + loss_s2_100 + loss_s2_075 + loss_s2_050 + loss_s3 + loss_s3_100 + loss_s3_075 + loss_s3_050
        reduced_loss =  loss_pose * pose_weight + loss_parsing * parsing_weight

        reduced_loss.backward()
        optimizer.step()

        if i%500==0 :
            print("[*] Loss EPOCH {} BATCH {}".format(e, i))
            print("[+] loss parsing  ", loss_parsing)
            print("[+] loss pose     ", loss_pose)
            print("[+] reduced loss  ", reduced_loss)


    torch.save({
        'ParsingNet_state_dict': parsing_net.state_dict(),
        'PoseNet_state_dict': pose_net.state_dict(),
        'PoseRefineNet_state_dict': pose_refine.state_dict(),
        'ParsingRefineNet_state_dict': parsing_refine.state_dict(),
        'BackboneResNet_state_dict': backbone.state_dict(),
        }, "./parameters/training_multi_epoch{}.pth".format(e))
    print("[!] Save Params epoch{}".format(e))
