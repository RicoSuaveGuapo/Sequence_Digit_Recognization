import torch
from torch.utils.data import *
from imutils import paths
import numpy as np
import random
import cv2
import os
import pandas as pd
import itertools

import matplotlib.pyplot as plt
from albumentations.augmentations.transforms import GaussNoise, Cutout, CoarseDropout, MultiplicativeNoise, RandomBrightness
from albumentations import Compose
# TODO: future need spatial transformation


CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'M', 'V', 'H','-'
        ]

# CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
#          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
#          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
#          'U', 'V', 'W', 'X', 'Y', 'Z'
#         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}


class net_dataset(Dataset):
    def __init__(self, list_path):
        super().__init__()
        with open(list_path, 'r') as f:
            self.img_files = f.readlines()

    def weighting(self):
        annotation_list = [d.strip().split(' ') for d in self.img_files]
        label_list      = [int(d[1]) for d in annotation_list]   
        w_class = []
        for i in range(0,2,1):
            class_i  = [int(d) for d in label_list if int(d) == i]
            w_class += [self.__len__()/len(class_i)]
        
        w_class = torch.tensor(w_class)
        
        return w_class

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        annotation = self.img_files[index].strip().split(' ')

        #---------
        #  Image
        #---------
        # NOTE: Notice the preprocess here, need to apply to later, too.
        img = cv2.imread(annotation[0],0)
        img = np.asarray(img, 'float32')
        img = (img - 37.7) / 255.
        input_img = torch.FloatTensor(img)
        input_img = input_img.unsqueeze(0)

        #---------
        #  Label
        #---------
        #  0: neg
        #  1: pos
        # -1: par
        label = int(annotation[1])
        bbox_target = np.zeros((4,))
        landmark = np.zeros((10,))
        
        if len(annotation[2:]) == 4:
            bbox_target = np.array(annotation[2:]).astype(float)
        sample = {'input_img': input_img, 'label': label, 'bbox_target': bbox_target}

        return sample



class LPRDataLoader(Dataset):
    def __init__(self, img_dir, df, mode, imgSize=(94, 24), PreprocFun=None):
        assert imgSize == (94, 24)
        assert os.path.isdir(img_dir), 'img_dir must be dir'
        assert mode.lower() in ['train','validation','test']

        mode_dict = {'train':0, 'validation':1, 'test':2}
        self.mode      = mode_dict[mode]
        self.data_mode = df['mode']
        index          = self.data_mode == self.mode

        self.img_dir   = img_dir
        self.name      = df['file_name'][index].tolist()
        self.img_paths = [os.path.join(self.img_dir, d) for d in self.name]
        self.img_size  = imgSize
        # TODO: For both
        # to account the upper and lower, need to repeat the path twice
        # self.img_paths = list(itertools.chain.from_iterable(itertools.repeat(d, 2) for d in self.img_paths))

        self.gt1       = df['GT_1'][index].tolist()
        self.gt2       = df['GT_2'][index].tolist()
        # TODO: For both
        # self.gt        = [self.gt1[int(i/2)] if i % 2 == 0 else self.gt2[i//2] for i in list(range(len(self.gt1)*2))]
        # TODO: For lower only
        # self.gt = df['GT_2'][index].tolist()
        # TODO: For upper only
        self.gt = df['GT_1'][index].tolist()
        
        self.xmin      = df['xmin'][index].tolist()
        self.ymin      = df['ymin'][index].tolist()
        self.xmax      = df['xmax'][index].tolist()
        self.ymax      = df['ymax'][index].tolist()
        self.ymax_2    = df['ymax_2'][index].tolist()

        # TODO: For both
        # self.xmin      = list(itertools.chain.from_iterable(itertools.repeat(d, 2) for d in self.xmin))
        # self.xmax      = list(itertools.chain.from_iterable(itertools.repeat(d, 2) for d in self.xmax))
        # self.ymin      = [self.ymin[int(i/2)] if i % 2 == 0 else self.ymax_2[i//2] for i in list(range(len(self.gt1)*2))]
        # self.ymax      = [self.ymax_2[int(i/2)] if i % 2 == 0 else self.ymax[i//2] for i in list(range(len(self.gt1)*2))]

        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        Image = cv2.imread(filename, 0)
        
        # since there are upper and lower bboxes,
        # even index will be upper, vice versa
        
        # TODO: For both
        # Image = Image[self.ymin[index]:self.ymax[index], self.xmin[index]:self.xmax[index]]
        # TODO: For lower only
        # Image = Image[self.ymax_2[index]:self.ymax[index], self.xmin[index]:self.xmax[index]]
        # TODO: For upper only
        Image = Image[self.ymin[index]:self.ymax_2[index], self.xmin[index]:self.xmax[index]]
        
        label = str(self.gt[index])

        height, width = Image.shape
        Image = cv2.resize(Image, self.img_size)
        # DANGER
        # cv2.imwrite('lprnet_original.jpg', Image)


        if self.mode == 0:
            Image = self.img_aug(Image)
        # DANGER
        # cv2.imwrite('lprnet_aug.jpg', Image)
        Image = self.PreprocFun(Image)        
        label, length = self.check(label)
        # len(label) gives the length of the label string
        return Image, label, length

    def transform(self, img):
        img = img.astype('float32')
        img = (img - 37.7) / 255.
        img = np.expand_dims(img, axis=0)
        return img
    
    def img_aug(self, img):
        transform = Compose([
                            CoarseDropout()
                            ])
        image_transform = transform(image = img)['image']

        return image_transform

    def split(self, word): 
        return [char for char in word] 

    def check(self, label):
        length = len(label)
        label_list = self.split(label)
        label_out  = []
        for char in label_list:
            if char in CHARS:
                label_out.append(CHARS_DICT[char])
            else:
                raise RuntimeError
        return label_out, length
        
        
def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)
        
if __name__ == "__main__":
    
    # dataset = LPRDataLoader(['validation'], (94, 24))   
    # dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2, collate_fn=collate_fn)
    # print('data length is {}'.format(len(dataset)))
    # for imgs, labels, lengths in dataloader:
    #     print('image batch shape is', imgs.shape)
    #     print('label batch shape is', labels.shape)
    #     print('label length is', len(lengths))      
    #     break
    
    # train_dataset = Pnet_dataset('anno_store/pnet_imglist_train.txt')
    # sample = next(iter(train_dataset))
    # print(sample)

    # LPRnet
    df = pd.read_csv('20201229_EXT_clear_2data_mode_resize.csv')
    lptdataset = LPRDataLoader(img_dir='data/20201229/EXT/resize', imgSize=(94, 24), df=df, mode='train')
    dataloader = DataLoader(lptdataset, batch_size=1, collate_fn=collate_fn)
    Image, label, length = next(iter(dataloader))
    # for i, data in enumerate(dataloader):
    #     if i < 10:
    #         print(data[1])
    #         print(data[2])
    #     else:
    #         break

