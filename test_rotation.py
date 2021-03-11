#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:07:10 2019

@author: xingyu
"""

from model import *
# from model.STN import STNet
from dataset import *
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import torchvision
import matplotlib.pyplot as plt
from itertools import groupby

def convert_image(inp):
    # convert a Tensor to numpy image
    inp = inp.numpy().transpose((1,2,0))
    inp = 127.5 + inp/0.0078125
    inp = inp.astype('uint8') 
    inp = inp[:,:,::-1]
    return inp

def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        dataset = LPRDataLoader([args.img_dirs], args.img_size)   
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn) 
        imgs, labels, lengths = next(iter(dataloader))
        
        input_tensor = imgs.cpu()
        transformed_input_tensor = STN(imgs.to(device)).cpu()
        
        in_grid = convert_image(torchvision.utils.make_grid(input_tensor))
        out_grid = convert_image(torchvision.utils.make_grid(transformed_input_tensor))
        
        # Plot the results side-by-side
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')
        
        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

# _, pred_labels = decode(preds, CHARS)
# preds: torch.Size([batch_size, CHARS length, output length ])
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

def eval(lprnet, dataloader, dataset, device, STN=None):
    
    lprnet = lprnet.to(device)
    if STN is not None:
        STN = STN.to(device)
    else:
        pass
    TP = 0
    for imgs, labels, lengths in dataloader:   # img: torch.Size([2, 3, 24, 94])  # labels: torch.Size([14]) # lengths: [7, 7] (list)
        imgs, labels = imgs.to(device), labels.to(device)
        if STN is not None:
            transfer = STN(imgs)
        else:
            transfer = imgs
        logits = lprnet(transfer) # torch.Size([batch_size, CHARS length, output length ])
    
        preds = logits.cpu().detach().numpy()  # (batch size, 68, 18)
        _, pred_labels = decode(preds, CHARS)  # list of predict output

        start = 0
        for i, length in enumerate(lengths):
            label = labels[start:start+length]
            start += length
            if np.array_equal(np.array(pred_labels[i]), label.cpu().numpy()):
                TP += 1
            
    ACC = TP / len(dataset) 
    
    return ACC
    

if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='LPR Evaluation')
#     parser.add_argument('--img_size', default=(94, 24), help='the image size')
#     parser.add_argument('--img_dirs', default="./data/ccpd_weather", help='the images path')
#     parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
#     parser.add_argument('--batch_size', default=128, help='batch size.')
#     args = parser.parse_args()
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     lprnet = LPRNet(class_num=len(CHARS), dropout_rate=args.dropout_rate)
#     lprnet.to(device)
#     lprnet.load_state_dict(torch.load('weights/Final_LPRNet_model.pth', map_location=lambda storage, loc: storage))
# #    checkpoint = torch.load('saving_ckpt/lprnet_Iter_023400_model.ckpt')
# #    lprnet.load_state_dict(checkpoint['net_state_dict'])
#     lprnet.eval() 
#     print("LPRNet loaded")
    
# #    torch.save(lprnet.state_dict(), 'weights/Final_LPRNet_model.pth')
    
#     STN = STNet()
#     STN.to(device)
#     STN.load_state_dict(torch.load('weights/Final_STN_model.pth', map_location=lambda storage, loc: storage))
# #    checkpoint = torch.load('saving_ckpt/stn_Iter_023400_model.ckpt')
# #    STN.load_state_dict(checkpoint['net_state_dict'])
#     STN.eval()
#     print("STN loaded")
    
# #    torch.save(STN.state_dict(), 'weights/Final_STN_model.pth')
    
#     dataset = LPRDataLoader([args.img_dirs], args.img_size)   
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn) 
#     print('dataset loaded with length : {}'.format(len(dataset)))
    
#     ACC = eval(lprnet, STN, dataloader, dataset, device)
#     print('the accuracy is {:.2f} %'.format(ACC*100))
    
#     visualize_stn()



    # ------------- My Test Section -------------
    # import pandas as pd
    # from dataset import *
    # from model import *
    # lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
    # lprnet.load_state_dict(torch.load('weights/lprnet/lprnet_92.77.pth', map_location=lambda storage, loc: storage))

    # df = pd.read_csv('20201229_EXT_clear_2data_mode_resize.csv')
    # train_dataset = LPRDataLoader(img_dir='data/20201229/EXT/resize', imgSize=(94, 24), df=df, mode='train')

    # train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn)
    # imgs, labels, lengths = next(iter(train_dataloader))
    # logits = lprnet(imgs)  # torch.Size([batch_size, CHARS length, output length ])
    # preds = logits.detach().numpy()  # (batch size, 14, 18)
    # _, pred_labels = decode(preds, CHARS)  # list of predict output
    # print(pred_labels)

    # ------ Rotation Testing ------
    # single image testing
    # model_path = 'weights/rotation/acc_100.00_loss_0.747.pth'
    # image_path = 'data/20201229/EXT/20201229101408_0EXT.bmp'
    # ori_image  = cv2.imread(image_path, 0)
    # image_size = 128
    # image = cv2.resize(ori_image, (image_size,image_size))
    # image = np.asarray(image, 'float32')
    # image = (image - 37.7) / 255.

    # image_list     = [image,
    #                 cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 
    #                 cv2.rotate(image, cv2.ROTATE_180), 
    #                 cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    # ori_image_list = [ori_image,
    #                 cv2.rotate(ori_image, cv2.ROTATE_90_CLOCKWISE), 
    #                 cv2.rotate(ori_image, cv2.ROTATE_180), 
    #                 cv2.rotate(ori_image, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    
    # image_tensors = [torch.tensor(image) for image in image_list]
    # image_tensors = [torch.reshape(image_tensor, (1,1,image_size,image_size)) for image_tensor in image_tensors] 

    # rotate_dict = {0:0, 1:90, 2:180, 3:270}

    # with torch.no_grad():
    #     rotation_model = Rotation_model(rgb=False, eval=True, img_size=image_size)
    #     rotation_model.load_state_dict(torch.load(model_path))
    #     rotation_model.eval()

    #     fig, axes = plt.subplots(2,2)
    #     for i in range(len(image_tensors)):
    #         rotate_key_prob = rotation_model(image_tensors[i])
    #         rotate_key_prob = rotate_key_prob.squeeze()
    #         rotate_key_prob = rotate_key_prob.numpy()
    #         # for i in range(4):
    #         #     print(f'Prob. of {rotate_dict[i]} deg  \t: {rotate_key_prob[i]:.3f}')
    #         preidiction = rotate_dict[np.argmax(rotate_key_prob)]
    #         # font = cv2.FONT_HERSHEY_SIMPLEX 
    #         # org = (0, ori_image.shape[0])
    #         # fontScale = 2
    #         # color = (0,255,0)
    #         # thickness = 2
    #         ori_image_list[i] = cv2.cvtColor(ori_image_list[i], cv2.COLOR_GRAY2RGB)
    #         # ori_image_list[i] = cv2.putText(ori_image_list[i], f'Prob. of {preidiction} deg: {np.max(rotate_key_prob):.3f}', 
    #         #                         org, font, fontScale, color, thickness, cv2.LINE_AA) 
    #         binindex = bin(i)
    #         if len(binindex) <= 3:
    #             axes[0,int(binindex[-1])].imshow(ori_image_list[i])
    #             axes[0,int(binindex[-1])].set_title(f'Prob. of {preidiction} deg: {np.max(rotate_key_prob):.3f}')
    #             axes[0,int(binindex[-1])].set_yticklabels([])
    #             axes[0,int(binindex[-1])].set_xticklabels([])
    #         else:
    #             axes[1,int(binindex[-1])].imshow(ori_image_list[i])
    #             axes[1,int(binindex[-1])].set_title(f'Prob. of {preidiction} deg: {np.max(rotate_key_prob):.3f}')
    #             axes[1,int(binindex[-1])].set_yticklabels([])
    #             axes[1,int(binindex[-1])].set_xticklabels([])
    #     fig.suptitle('Testing set image', fontsize=16)
    #     plt.show()

    # testing set testing
    criterion = nn.CrossEntropyLoss()
    model_path     = 'weights/rotation/acc_100.00_loss_0.747.pth'
    image_size     = 128
    test_dataset   = RotationDataset(path='data/20201229/EXT/resize',mode='test', img_size=image_size)
    test_loader    = DataLoader(test_dataset,  batch_size=128, shuffle=False, num_workers=2*os.cpu_count(), pin_memory=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        rotation_model = Rotation_model(rgb=False, eval=True, img_size=image_size)
        rotation_model.load_state_dict(torch.load(model_path))
        rotation_model.eval()
        rotation_model.to(device)
        test_loss     = 0.0
        batch_count   = 0
        total_count   = 0
        correct_count = 0
        for data in test_loader:
            imgs, labels = data[0].to(device), data[1].to(device)
            outputs      = rotation_model(imgs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            test_loss   += loss.item()
            correct_count += (predicted == labels).sum().item()
            batch_count += 1
            total_count += labels.size(0)
        accuracy = (100 * correct_count/total_count)
        test_loss = test_loss/batch_count
    print(f'Accuracy on testing set: {accuracy:.2f}%')
    print(f'Total images count     : {len(test_dataset)}')
    print(f'Loss on testing set    : {test_loss:.2f}')
