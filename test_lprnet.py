#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:49:57 2019

@author: xingyu
"""
import sys
import os
from PIL import Image, ImageDraw, ImageFont
from model import *
import numpy as np
import argparse
import torch
import time
import cv2

import matplotlib.pyplot as plt

CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'M', 'V', 'H','-'
        ]
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

def convert_image(inp):
    # convert a Tensor to numpy image
    inp = inp.squeeze(0).cpu()
    inp = inp.detach().numpy().transpose((1,2,0))
    inp = 127.5 + inp/0.0078125
    inp = inp.astype('uint8') 

    return inp

def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# def decode(preds, CHARS):
#     # greedy decode
#     pred_labels = list()
#     labels = list()
#     for i in range(preds.shape[0]):
#         pred = preds[i, :, :]
#         pred_label = list()
#         for j in range(pred.shape[1]):
#             pred_label.append(np.argmax(pred[:, j], axis=0))
#         no_repeat_blank_label = list()
#         pre_c = pred_label[0]
#         for c in pred_label: # dropout repeate label and blank label
#             if (pre_c == c) or (c == len(CHARS) - 1):
#                 if c == len(CHARS) - 1:
#                     pre_c = c
#                 continue
#             no_repeat_blank_label.append(c)
#             pre_c = c
#         pred_labels.append(no_repeat_blank_label)
        
#     for i, label in enumerate(pred_labels):
#         lb = ""
#         for i in label:
#             lb += CHARS[i]
#         labels.append(lb)
    
#     return labels, np.array(pred_labels)  

def decode(preds, CHARS):
    # greedy decode
    pred_labels = list()
    labels      = list()
    for i in range(preds.shape[0]):
        pred = preds[i, :, :]
        # torch.Size([CHARS length: 14, output length: 18 ])
        pred_label            = list()
        no_repeat_blank_label = list()
        
        # greedy decode here
        for j in range(pred.shape[1]):
            pred_label.append(np.argmax(pred[:, j], axis=0))
        
        # print(pred_label)
        # print(len(pred_label))
        # [13, 13, 1, 13, 13, 0, 0, 0, 0, 13, 2, 13, 13, 13, 10, 10, 13, 11]
        # 18

        # author implemetation
        # pre_c = pred_label[0] # for example: 3
        # for c in pred_label: # dropout repeate label and blank label
        #     if (pre_c == c) or (c == len(CHARS) - 1):
        #         if c == len(CHARS) - 1:
        #             pre_c = c
        #             # pre_c = 13
        #         continue
        #     no_repeat_blank_label.append(c)
        #     pre_c = c
        # pred_labels.append(no_repeat_blank_label)

        # My implementation
        # 1. remove the duplicate (not including the blank)
        blank = CHARS_DICT['-']
        output= []
        for i, d in enumerate(pred_label):
            if d == blank:
                output.append(d)
            else:
                if pred_label[i] == pred_label[i+1:i+2]:
                    pass
                else:
                    output.append(d)
        # 2. remove the blank
        output = [d for d in output if d != blank]
        pred_labels.append(output)

    for i, label in enumerate(pred_labels):
        lb = ""
        for i in label:
            lb += CHARS[i]
        labels.append(lb)
    
    return labels, pred_labels

if __name__ == '__main__':
    
    save_path = 'tmp_result/LPRnet_result'

    if os.path.exists(os.path.join(save_path, 'correct')):
        shutil.rmtree(os.path.join(save_path, 'correct'))
        shutil.rmtree(os.path.join(save_path, 'wrong'))
        os.mkdir(os.path.join(save_path, 'correct'))
        os.mkdir(os.path.join(save_path, 'wrong'))
    elif not os.path.exists(os.path.join(save_path, 'correct')):
        os.mkdir(os.path.join(save_path, 'correct'))
        os.mkdir(os.path.join(save_path, 'wrong'))


    CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'M', 'V', 'H','-'
            ]
    CHARS_reverse = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9,
                     '10':'M', '11':'V', '12':'H','13':'-'}

    parser = argparse.ArgumentParser(description='LPR Result Demo')
    parser.add_argument("--mode", help='0: train, 1: val, 2: test')
    parser.set_defaults(mode=1)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lprnet_up = LPRNet(class_num=len(CHARS), dropout_rate=0)
    lprnet_do = LPRNet(class_num=len(CHARS), dropout_rate=0)
    lprnet_up.to(device)
    lprnet_do.to(device)
    lprnet_up.load_state_dict(torch.load('tmp_result/LPRnet_result/reserved_weight/upper_96.39.pth'))
    lprnet_do.load_state_dict(torch.load('tmp_result/LPRnet_result/reserved_weight/lower_98.80.pth'))
    lprnet_up.eval()
    lprnet_do.eval()
    print("Successful to build network!")
    
    df         = pd.read_csv('data.csv')
    image_path = 'data/20201229/EXT/resize/new_image'
    
    data_mode = df['mode']
    index     = data_mode == args.mode
    name      = df['file_name'][index].tolist()
    img_paths = [os.path.join(image_path, d) for d in name]
    
    xmin      = df['xmin'][index].tolist()
    ymin      = df['ymin'][index].tolist()
    xmax      = df['xmax'][index].tolist()
    ymax      = df['ymax'][index].tolist()
    ymax_2    = df['ymax_2'][index].tolist()
    gt_1      = df['GT_1'][index].tolist()
    gt_2      = df['GT_2'][index].tolist()

    enlarge = 10
    xmin      = [d - enlarge for d in xmin]
    xmax      = [d + enlarge for d in xmax]

    gt_1 = [str(d) for d in gt_1]
    gt   = [d+gt_2[i] for i, d in enumerate(gt_1)]

    images_ori = [cv2.imread(d,0) for d in img_paths if d.endswith('.bmp')]
    # for upper region
    upper     = [d[ymin[i]:ymax_2[i], xmin[i]:xmax[i]] for i, d in enumerate(images_ori)]
    # for lower region
    lower     = [d[ymax_2[i]:ymax[i], xmin[i]:xmax[i]] for i, d in enumerate(images_ori)]
    
    upper_res = [cv2.resize(d, (94, 24), interpolation=cv2.INTER_CUBIC) for d in upper]
    lower_res = [cv2.resize(d, (94, 24), interpolation=cv2.INTER_CUBIC) for d in lower]
    # image = cv2.imread(args.image,0)   
    
    # im = cv2.resize(image, (94, 24), interpolation=cv2.INTER_CUBIC)
    # im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5)*0.0078125
    # data = torch.from_numpy(im).float().unsqueeze(0).to(device)  # torch.Size([1, 3, 24, 94]) 
    
    # im = np.asarray(im, 'float32')
    # im = (im - 37.7) / 255.
    # input_img = torch.FloatTensor(im)
    # input_img = input_img.unsqueeze(0)
    # input_img = input_img.unsqueeze(0).to(device)

    upper_res = [(np.asarray(im, 'float32')- 37.7) / 255. for im in upper_res]
    lower_res = [(np.asarray(im, 'float32')- 37.7) / 255. for im in lower_res]
    upper_res = [torch.FloatTensor(im).unsqueeze(0).unsqueeze(0).to(device) for im in upper_res]
    lower_res = [torch.FloatTensor(im).unsqueeze(0).unsqueeze(0).to(device) for im in lower_res]

    acc = 0
    for idx, upper in enumerate(upper_res):
        preds_up = lprnet_up(upper)
        preds_do = lprnet_do(lower_res[idx])

        preds_up = preds_up.cpu().detach().numpy()  # (1, 68, 18)
        preds_do = preds_do.cpu().detach().numpy()  # (1, 68, 18)
        since = time.time()
        labels_up, pred_labels_up = decode(preds_up, CHARS)
        labels_do, pred_labels_do = decode(preds_do, CHARS)

        if idx == len(upper)-1:
            print("model inference in {:.2E} seconds".format(time.time() - since))

        pred_labels_up = [CHARS_reverse[f'{d}'] for d in pred_labels_up[0]]
        pred_labels_up = ''.join([str(d) for d in pred_labels_up])

        pred_labels_do = [CHARS_reverse[f'{d}'] for d in pred_labels_do[0]]
        pred_labels_do = ''.join([str(d) for d in pred_labels_do])

        pred_labels = pred_labels_up + pred_labels_do

        if pred_labels == gt[idx]:
            acc += 1
            img = cv2ImgAddText(images_ori[idx], pred_labels, (0, 630), textColor=(0,255,0), textSize=100)
            cv2.imwrite(f'tmp_result/LPRnet_result/correct/{os.path.basename(img_paths[idx])}', img)
        else:
            img = cv2ImgAddText(images_ori[idx], pred_labels, (0, 630), textColor=(255,0,0), textSize=100)
            img = cv2ImgAddText(img, gt[idx], (0, 530), textColor=(0,0,255), textSize=100)
            cv2.imwrite(f'tmp_result/LPRnet_result/wrong/{os.path.basename(img_paths[idx])}', img)
        if idx % 10 == 0 and idx != 0:
            print(f'{idx} images are saved')
    
    acc = acc/len(upper_res)
    if args.mode == 0:
        print('\nIn Training set')
        print(f'Accuracy is {acc*100:.2f} %')
    elif args.mode == 1:
        print('\nIn Validation set')
        print(f'Accuracy is {acc*100:.2f} %')
    else:
        print('\nIn Testing set')
        print(f'Accuracy is {acc*100:.2f} %')
    # transformed_img = convert_image(transfer)
    # cv2.imshow('transformed', transformed_img)
    
    # plt.imshow(img)
    # plt.show()
    # cv2.imshow("test", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    