import argparse
import os
import time
from decimal import Decimal
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from ranger import Ranger
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts)
from torch.utils.data import DataLoader

from dataset import *
from Evaluation import decode, eval
from model import *
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rotation Training')
    parser.add_argument('--img_dir', help='location of images')
    parser.add_argument('--img_size', help='image size')
    parser.add_argument('--epoch', type=int, default=33, help='number of epoches for training')
    parser.add_argument('--pth', help='the previous trained model weights')
    parser.add_argument('--batch_size', default=16, help='batch size')
    parser.add_argument('--save_dir', help='location to save .pth')
    parser.set_defaults(epoch=50, img_size=128, batch_size=64, img_dir='data/20201229/EXT/resize', \
                        save_dir='weights/rotation')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # model
    rotation_model = Rotation_model(rgb=False, img_size=args.img_size)
    rotation_model.to(device)
    print("model loaded")
    if args.pth is not None:
        rotation_model.load_state_dict(torch.load(args.pth, map_location=lambda storage, loc: storage))
        print('pre-trained model loaded')
    
    train_dataset = RotationDataset(path=args.img_dir,mode='train', img_size=args.img_size)
    val_dataset   = RotationDataset(path=args.img_dir,mode='val', img_size=args.img_size)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    val_dataloader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=True, num_workers=2*os.cpu_count(), pin_memory=True)
    print('training dataset loaded with length : {}'.format(len(train_dataset)))
    print('validation dataset loaded with length : {}'.format(len(val_dataset)))
    
    # define optimizer & loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rotation_model.parameters())

    # NOTE:
    # scheduler = build_scheduler(optimizer, 'ReduceLROnPlateau')
    # steps = 10
    # scheduler = CosineAnnealingLR(optimizer, steps)
    # T_0 = 400
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0)

    scheduler = ReduceLROnPlateau(optimizer, mode='min')
    

    # save logging and weights
    # if not os.path.exists('log'):
    #     os.mkdir('log')
    # # training log file
    # train_logging_file = 'log/rotation_model.txt'
    # if os.path.exists(train_logging_file):
    #     os.remove(train_logging_file)
    # # validation log file
    # validation_logging_file = 'log/rotation_model.txt'
    # if os.path.exists(validation_logging_file):
    #     os.remove(validation_logging_file)
    
    start_time = time.time()
    best_acc = [0.]
    best_val_loss = [10.]
    total_iters = 0
    for epoch in range(args.epoch):
        # train model
        print(f'Epoch {epoch+1}:')
        rotation_model.train()
        # print('--- Training Loop Begins ---')
        for imgs, labels in train_dataloader: 
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            total_iters += 1
            
            outputs = rotation_model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if total_iters % 2 == 0:
                _, predicted = torch.max(outputs, 1)
                TP           = (predicted == labels).sum().item()
                total        = labels.size(0)

                for p in  optimizer.param_groups:
                    lr = p['lr']
                print("Epoch {}/{}, train_loss: {:.4f}, train_accuracy: {:.2f}%, learning rate: {:.2E}"
                        .format(epoch+1, args.epoch, loss.item(), TP/total*100, Decimal(lr)))
                
        with torch.no_grad():
            rotation_model.eval()
            val_run_loss = 0.0
            batch_count  = 0
            total_count   = 0
            correct_count = 0

            for data in val_dataloader:
                imgs, labels = data[0].to(device), data[1].to(device)
                outputs      = rotation_model(imgs)
                _, predicted = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_run_loss  += loss.item()
                correct_count += (predicted == labels).sum().item()
                batch_count += 1
                total_count += labels.size(0)
            accuracy = (100 * correct_count/total_count)
            val_run_loss = val_run_loss/batch_count
        
        if max(best_acc) >= 80:
            scheduler.step(val_run_loss)

        if max(best_acc) <= accuracy and min(best_val_loss) > val_run_loss:
            torch.save(rotation_model.state_dict(), os.path.join(args.save_dir, f'acc_{accuracy:.2f}_loss_{val_run_loss:.3f}.pth'))
            print('\n-------- Saveing the best weight --------')
        else:
            pass
            # print('-------- Accuracy is not improving --------\n')
        best_acc.append(accuracy)
        best_val_loss.append(val_run_loss)

        # for ReduceLROnPlateau
        scheduler.step(val_run_loss)
        # CosineAnnealingWarmRestarts
        # scheduler.step(epoch + total_iters / len(train_dataloader))

        for p in  optimizer.param_groups:
                lr = p['lr']
        print("\nEpoch {}/{}, valid_loss: {:.4f}, valid_accuracy: {:.2f}%, learning rate: {:.2E}"
                .format(epoch+1, args.epoch, val_run_loss, accuracy, Decimal(lr)))

    time_elapsed = time.time() - start_time
    print('-'*10)
    print('\nFinal Best Accuracy: {:.2f}%'.format(max(best_acc)))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
