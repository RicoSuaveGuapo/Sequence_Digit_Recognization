import os
import sys
import pandas as pd
import shutil
from MTCNN_evaluation import *


def gt_creator(df,des_path):
    assert des_path.endswith('groundtruths'), 'Create des_path must ends with "groundtruths"'
    if not os.path.exists(des_path):
        os.mkdir(des_path)

    file_name = df['file_name'].tolist()
    file_name = [os.path.splitext(d)[0] for d in file_name]
    file_name = [d+'.txt' for d in file_name]

    cls_name  = 'Digits'
    xmin_1    = df['xmin'].tolist()
    ymin_1    = df['ymin'].tolist()
    xmax_1    = df['xmax'].tolist()
    ymax_1    = df['ymax'].tolist()

    xmin_2    = df['xmin'].tolist()
    ymin_2    = df['ymax_2'].tolist()
    xmax_2    = df['xmax'].tolist()
    ymax_2    = df['ymax'].tolist()

    for i, name in enumerate(file_name):
        with open(os.path.join(des_path, name), 'w') as f:
            f.write(f'{cls_name} {xmin_1[i]} {ymin_1[i]} {xmax_1[i]} {ymax_1[i]}\n')
            f.write(f'{cls_name} {xmin_2[i]} {ymin_2[i]} {xmax_2[i]} {ymax_2[i]}\n')

        if i % 50 == 0:
            print(f'{i} gt files created')

def pred_creator(img_dir, des_path):
    assert des_path.endswith('detections'), 'Create des_path must ends with "detections"'
    if not os.path.exists(des_path):
        os.mkdir(des_path)
    else:
        shutil.rmtree(des_path)
        os.mkdir(des_path)

    # output_prob must be true to be activated
    mtcnn_visual(img_dir = img_dir, output_dir=des_path, output_prob=True)


if __name__ == "__main__":
    # Creation of gt .txts
    # df = pd.read_csv('20201229_EXT_clear_2data_mode_resize.csv')
    # gt_creator(df, 'data/20201229/EXT/groundtruths')
    # Creation of pred .txts
    pred_creator(img_dir='data/20201229/EXT/resize', des_path='data/20201229/EXT/detections')

    Iou_threshold  = 0.3
    save_plot_path = '/home/rico-li/Job/Feng_Hsin_steel/map_plot_save' 
    if not os.path.exists(save_plot_path):
        os.mkdir(save_plot_path)

    sys.path.append('/home/rico-li/Job/Object-Detection-Metrics')
    os.system(f'python /home/rico-li/Job/Object-Detection-Metrics/pascalvoc.py \
              -t {Iou_threshold} \
              -sp {save_plot_path}  \
              -gtformat xyrb -detformat xyrb \
              --gt /home/rico-li/Job/Feng_Hsin_steel/data/20201229/EXT/groundtruths \
              --det /home/rico-li/Job/Feng_Hsin_steel/data/20201229/EXT/detections')
    print(f'plot is saved in {save_plot_path}')

