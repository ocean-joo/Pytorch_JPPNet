import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

BASE_DIR = "/mnt/ssd/Data/LIP/"

DATA_DIR = BASE_DIR + 'TrainVal_images/train_images/'
SEGMENTATION_PATH = BASE_DIR + "TrainVal_parsing_annotations/train_segmentations/"
REVERSE_DIR = BASE_DIR + "train_segmentations_reversed/"
POSE_DIR = BASE_DIR + 'TrainVal_pose_annotations/heatmap/'
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
NUM_POSE = 16


class myTensor(object) :
    def __call__(self, pic) :
        if isinstance(pic, np.ndarray) :
            pic_r, pic_g, pic_b = np.split(pic, 3, axis=2)
            pic = np.concatenate([pic_b, pic_g, pic_r], axis=2)
            pic = pic - IMG_MEAN
            pic = torch.from_numpy(pic).permute(2,0,1)
            return pic
        elif Image.isImageType(pic) :
            pic = np.array(pic)
            pic_r, pic_g, pic_b = np.split(pic, 3, axis=2)
            pic = np.concatenate([pic_b, pic_g, pic_r], axis=2)
            pic = pic - IMG_MEAN
            pic = torch.from_numpy(pic).permute(2,0,1)
            return pic
        else : # error
            raise TypeError

class LIPDataset(Dataset) :
    def __init__(self, base_dir, transform=None) :
        self.base_dir = base_dir
        self.data_dir = self.base_dir + 'TrainVal_images/train_images/'
        self.seg_dir = self.base_dir + "TrainVal_parsing_annotations/train_segmentations/"
        self.rev_dir = self.base_dir + "train_segmentations_reversed/"
        self.pose_dir = self.base_dir + 'TrainVal_pose_annotations/heatmap/'
        self.train_id = []
        with open(self.base_dir+'TrainVal_images/train_id.txt', 'r') as f:
            lines = f.readlines()
            for line in lines :
                self.train_id.append(line.strip())
        self.transforms = transform

    def __len__(self) :
        return len(self.train_id)

    def __getitem__(self, index) :
        image_id = self.train_id[index]

        ### 이부분 transform으로 넣기
        image = Image.open(self.data_dir + image_id + '.jpg')
        segment = Image.open(self.seg_dir + image_id + '.png')
        segment_rev = Image.open(self.rev_dir + image_id + '.png')

        pose = []
        for i in range(NUM_POSE) :
            pose_contents = Image.open(self.pose_dir + image_id + '_{}.png'.format(i))
            pose.append(np.array(pose_contents))
        heatmap = torch.from_numpy(np.stack(pose, axis=2))

        pose_rev = [None] * 16
        pose_rev[0] = pose[5]
        pose_rev[1] = pose[4]
        pose_rev[2] = pose[3]
        pose_rev[3] = pose[2]
        pose_rev[4] = pose[1]
        pose_rev[5] = pose[0]
        pose_rev[10] = pose[15]
        pose_rev[11] = pose[14]
        pose_rev[12] = pose[13]
        pose_rev[13] = pose[12]
        pose_rev[14] = pose[11]
        pose_rev[15] = pose[10]
        pose_rev[6] = pose[6]
        pose_rev[7] = pose[7]
        pose_rev[8] = pose[8]
        pose_rev[9] = pose[9]

        heatmap_rev = np.stack(pose_rev, axis=2)
        heatmap_rev = np.flip(heatmap_rev, axis=2)

        if self.transforms :
            image = self.transforms(image)
            h, w = image.size
            segment = np.array(segment.resize((h, w)))
            heatmap = heatmap.permute(2,0,1).unsqueeze(0).float()
            heatmap = F.interpolate(heatmap, size=(h, w)).squeeze(0)

        image = np.array(image)
        segment = torch.from_numpy(np.array(segment))
        segment_rev = torch.from_numpy(np.array(segment_rev))
        return image, segment, heatmap
