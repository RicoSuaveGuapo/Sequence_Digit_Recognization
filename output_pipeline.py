from PIL import Image
import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# rotation and LPRnet models
from model import *
from dataset import *
import torch
from torch.utils.data import DataLoader

# YOLO
# sys.path.append('/home/rico-li/Job/yolov4-pytorch/')
# sys.path.append('/home/rico-li/Job/yolov4-pytorch/nets/')
# from ..yolov4-pytorch import *
# from yolo import YOLO


if __name__ == '__main__':
    target_img_path = 'data/20201229/EXT/resize/20201229080315_0EXT.bmp'
    # for Rotation Head
    ori_image_rot   = cv2.imread(target_img_path, 0)
    # for YOLO
    image_yolo = Image.open(target_img_path)
    image_yolo = image_yolo.convert('RGB')

    # ==================== Rotation ====================
    model_path = 'weights/rotation/acc_100.00_loss_0.747.pth'
    image_size = 128
    image = cv2.resize(ori_image_rot, (image_size,image_size))
    image = np.asarray(image, 'float32')
    image = (image - 37.7) / 255.
   
    image_tensor = torch.tensor(image)
    image_tensor = torch.reshape(image_tensor, (1,1,image_size,image_size))

    rotate_dict = {0:'Normal', 1:'90 deg', 2:'180 deg', 3:'270 deg'}

    with torch.no_grad():
        rotation_model = Rotation_model(rgb=False, eval=True, img_size=image_size)
        rotation_model.load_state_dict(torch.load(model_path))
        rotation_model.eval()

        rotate_key_prob = rotation_model(image_tensor).squeeze().numpy()
        preidiction     = rotate_dict[np.argmax(rotate_key_prob)]
    
    if preidiction == 'Normal':
        rot_image = image_yolo
        print(f'Image orientation is {preidiction}')
    elif preidiction == '90 deg':
        # PIL rotate is in counter clockwise
        rot_image = image_yolo.rotate(270)
        print(f'Image is {preidiction}')
    elif preidiction == '180 deg':
        rot_image = image_yolo.rotate(180)
        print(f'Image is {preidiction}')
    else:
        rot_image = image_yolo.rotate(90)
        print(f'Image is {preidiction}')
    
    print('Rotation model Done')
    
    # ==================== Detection ====================
    # yolo = YOLO()
    # note that the weight of the YOLO is input in 
    # /home/rico-li/Job/yolov4-pytorch/yolo.py
    # modify it if needed

    # result  = yolo.detect_image(rot_image)
    # result  = np.array(result)
    # plt.imshow(r_image)
    # plt.show()

    # ==================== Recognization ====================
