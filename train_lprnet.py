import argparse
import os
import time
from decimal import Decimal

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from ranger import Ranger
from torch.utils.data import DataLoader
# TODO
# from cutmix.cutmix import CutMix

from dataset import *
from Evaluation import decode, eval
from model import *
from utils import *

# CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
#          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
#          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
#          'U', 'V', 'W', 'X', 'Y', 'Z'
#         ]
CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'M', 'V', 'H','-'
        ]

# sparse_tuple_for_ctc(T_length, lengths)
def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)

if __name__ == '__main__':

    # LPRNet_model_Init_pth = '/home/rico-li/Job/Object_Detection/License_Plate_Detection_Pytorch/LPRNet/weights/LPRNet_model_Init.pth'

    parser = argparse.ArgumentParser(description='LPR Training')
    parser.add_argument('--img_dir', help='location of images')
    parser.add_argument('--df', help='dataframe')
    parser.add_argument('--img_size', default=(94, 24), help='the image size')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--epoch', type=int, default=33, help='number of epoches for training')
    parser.add_argument('--pth', help='the trained model weights, the --part must be matched')
    parser.add_argument('--batch_size', default=16, help='batch size')
    parser.add_argument('--save_dir', help='batch size')
    parser.add_argument('--part', help='which part should be trained')
    parser.set_defaults(epoch=1024, batch_size=16, img_dir='data/20201229/EXT/resize', \
                        df='data.csv', save_dir='weights/lprnet', \
                        pth='tmp_result/LPRnet_result/reserved_weight/lower_95.18.pth', part='lower')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # model
    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=args.dropout_rate, color=False)
    lprnet.to(device)
    print("LPRNet loaded")
    if args.pth is not None:
        lprnet.load_state_dict(torch.load(args.pth, map_location=lambda storage, loc: storage))
        print('pre-trained model loaded')
        if os.path.basename(args.pth).startswith('lower'):
            print('\n ----Load in lower part pre-trained----')
            assert args.part == 'lower'
        elif os.path.basename(args.pth).startswith('upper'):
            print('\n ----Load in upper part pre-trained----')
            assert args.part == 'upper'
        else:
            raise AssertionError('make sure that the pre-trained model has corresponding data, use --part to modified')
    
    df = pd.read_csv(args.df)
    train_dataset = LPRDataLoader(img_dir=args.img_dir, imgSize=args.img_size, df=df, mode='train', part=args.part)
    val_dataset   = LPRDataLoader(img_dir=args.img_dir, imgSize=args.img_size, df=df, mode='validation', part=args.part)

    # TODO
    # Cutmix
    # train_dataset = CutMix(train_dataset, num_class=)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True, collate_fn=collate_fn)
    val_dataloader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=True, num_workers=2*os.cpu_count(), pin_memory=True, collate_fn=collate_fn)
    print('training dataset loaded with length : {}'.format(len(train_dataset)))
    print('validation dataset loaded with length : {}'.format(len(val_dataset)))
    
    # define optimizer & loss
    optimizer = torch.optim.Adam(lprnet.parameters())

    # optimizer  = Ranger(lprnet.parameters(), lr = 0.001)

    ctc_loss  = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'

    # NOTE:
    # scheduler = build_scheduler(optimizer, 'ReduceLROnPlateau')
    # steps = 10
    # scheduler = CosineAnnealingLR(optimizer, steps)
    T_0 = 400
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0)

    ## save logging and weights
    if not os.path.exists('log'):
        os.mkdir('log')
    # training log file
    train_logging_file = 'log/lprnet_train_log.txt'
    if os.path.exists(train_logging_file):
        os.remove(train_logging_file)
    # validation log file
    validation_logging_file = 'log/lprnet_val_log.txt'
    if os.path.exists(validation_logging_file):
        os.remove(validation_logging_file)
    
    start_time = time.time()
    total_iters = 0
    best_acc = [0.]
    T_length = 18 # args.lpr_max_len
    print('training kicked off..')
    print('-' * 10) 
    for epoch in range(args.epoch):
        # train model
        lprnet.train()
        since = time.time()
        for imgs, labels, lengths in train_dataloader:   # img: torch.Size([2, 3, 24, 94])  # labels: torch.Size([14]) # lengths: [7, 7] (list)
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logits = lprnet(imgs)  # torch.Size([batch_size, CHARS length, output length ])
            # print(logits.shape)
            # torch.Size([16, 14, 18])
            log_probs = logits.permute(2, 0, 1) # for ctc loss: length of output x batch x length of chars
            # torch.Size([18, 16, 14])
            log_probs = log_probs.log_softmax(2).requires_grad_()    
            input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths) # convert to tuple with length as batch_size 
            
            loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
            # Log_probs:
            #     Tensor of size (T, N, C)
            #     where T =  input length, N = batch size, and C = number of classes (including blank)}
            # Targets:
            #     Tensor of size (N, S) or (sum(target_lengths))
            # Input_lengths:
            #     Tuple or tensor of size (N), where N = batch size.
            # Target_lengths
            #     Tuple or tensor of size (N), where N = batch size.

            loss.backward()
            optimizer.step()
            
            total_iters += 1
            # print train information
            if total_iters % 100 == 0:
                # current training accuracy             
                preds = logits.cpu().detach().numpy()  # (batch size, 14, 18)
                _, pred_labels = decode(preds, CHARS)  # list of predict output

                total = preds.shape[0]
                start = 0
                TP = 0
                for i, length in enumerate(lengths):
                    label = labels[start:start+length]
                    start += length
                    if np.array_equal(np.array(pred_labels[i]), label.cpu().numpy()):
                        TP += 1

                time_cur = (time.time() - since) / 100
                since = time.time()
                
                for p in  optimizer.param_groups:
                    lr = p['lr']
                print("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.2f}%, time: {:.2f} s/iter, learning rate: {:.2E}"
                        .format(epoch+1, args.epoch, total_iters, loss.item(), TP/total*100, time_cur, Decimal(lr)))
                with open(train_logging_file, 'a') as f:
                    f.write("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.2f}%, time: {:.2f} s/iter, learning rate: {}"
                        .format(epoch+1, args.epoch, total_iters, loss.item(), TP/total*100, time_cur, Decimal(lr))+'\n')
                f.close()
                
            if total_iters % 400 == 0:
                # evaluate accuracy
                lprnet.eval()
                with torch.no_grad():
                    ACC = eval(lprnet, val_dataloader, val_dataset, device)

                if ACC >= max(best_acc):
                    # save model
                    if args.part == 'lower':
                        torch.save(lprnet.state_dict(), os.path.join(args.save_dir, f'lower_{ACC*100:.2f}.pth'))
                    elif args.part == 'upper':
                        torch.save(lprnet.state_dict(), os.path.join(args.save_dir, f'upper_{ACC*100:.2f}.pth'))
                    print('\n-------- Saveing the best weight --------')
                else:
                    print('\n-------- Accuracy is not improving --------')
                best_acc.append(ACC)
                
                # scheduler
                # for ReduceLROnPlateau
                # scheduler.step(ACC)
                # for CosineAnnealingLR
                # scheduler.step()

                print("Epoch {}/{}, Iters: {:0>6d}, validation_accuracy: {:.2f}%\n".format(epoch+1, args.epoch, total_iters, ACC*100))
                with open(validation_logging_file, 'a') as f:
                    f.write("Epoch {}/{}, Iters: {:0>6d}, validation_accuracy: {:.2f}%".format(epoch+1, args.epoch, total_iters, ACC*100)+'\n')
                f.close()
                
                lprnet.train()
            
            # for CosineAnnealingWarmRestarts
            scheduler.step(epoch + total_iters / len(train_dataloader))
        
        # for CosineAnnealingLR
        # scheduler = CosineAnnealingLR(optimizer, steps)
                                
    time_elapsed = time.time() - start_time  
    print('-'*10)
    print('\nFinal Best Accuracy: {:.2f}%'.format(max(best_acc)*100))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
