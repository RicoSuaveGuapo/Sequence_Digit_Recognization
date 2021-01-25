import argparse
import time
import os

import cv2
import matplotlib.pyplot as plt

from model import *

# TODO:
# def mtcnn_metric():
#     pass


def mtcnn_visual(img_path:str=None, img_dir:str=None, output_dir:str=None, output_prob=False, prob_thresholds=0.8, output=False):
    parser = argparse.ArgumentParser(description='MTCNN Evalution')
    parser.add_argument("--test_image", dest='test_image', help=
    "test image path", default=img_path, type=str)
    parser.add_argument("--test_dir", dest='test_dir', help=
    "test image dir", default=img_dir, type=str)
    parser.add_argument("--scale", dest='scale', help=
    "scale the iamge", default=1, type=int)
    parser.add_argument('--mini_lp', dest='mini_lp', help=
    "Minimum lp to be detected. derease to increase accuracy. Increase to increase speed",
                        default=(50, 15), type=int)

    args = parser.parse_args()
    assert not (img_path is not None and img_dir is not None), 'only one of them can be used'
    assert img_dir is not None and output_dir is not None, 'if given img_dir, output_dir also needed'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if img_path is not None:
        image = cv2.imread(args.test_image,0)
        image = cv2.resize(image, (0, 0), fx = args.scale, fy = args.scale, interpolation=cv2.INTER_CUBIC)

        start = time.time()

        bboxes = create_mtcnn_net(image, args.mini_lp, device, p_model_path='weights/pnet/pnet_best.pth', o_model_path='weights/onet/Onet_best.pth')

        print("image predicted in {:2.3f} seconds".format(time.time() - start))

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, :4]
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 255, 2)
            
        image = cv2.resize(image, (0, 0), fx = 1/args.scale, fy = 1/args.scale, interpolation=cv2.INTER_CUBIC)
        plt.imshow(image)
        plt.show()

    elif img_dir is not None:
        img_paths = os.listdir(args.test_dir)
        img_paths = [os.path.join(args.test_dir, d) for d in img_paths if d.endswith('.bmp')]
        img_list  = [cv2.imread(d,0) for d in img_paths]

        img_list  = [cv2.resize(d, (0, 0), fx = args.scale, fy = args.scale, \
                    interpolation=cv2.INTER_CUBIC) for d in img_list]

        if not output_prob:
            if not output:
                # check the label region
                bboxes_list        = create_mtcnn_net_list(img_list, args.mini_lp, device, \
                                                    p_model_path='weights/pnet/pnet_best.pth', \
                                                    o_model_path='weights/onet/Onet_best.pth', prob_thresholds=prob_thresholds)
                for idx, bboxes in enumerate(bboxes_list):
                    for i in range(bboxes.shape[0]):
                        bbox = bboxes[i, :4]
                        cv2.rectangle(img_list[idx], (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 255, 2)
                    img_list[idx] = cv2.resize(img_list[idx], (0, 0), fx = 1/args.scale, fy = 1/args.scale, interpolation=cv2.INTER_CUBIC)
                
                action = [cv2.imwrite(os.path.join(output_dir, os.path.basename(img_paths[i])), d) for i, d in enumerate(img_list)]

                return img_list
            
            elif output:
                # for the use of LPRnet
                prob_thresholds = 0.5
                bboxes_list        = create_mtcnn_net_list(img_list, args.mini_lp, device, \
                                                    p_model_path='weights/pnet/pnet_best.pth', \
                                                    o_model_path='weights/onet/Onet_best.pth', prob_thresholds=prob_thresholds)
                output_list = []
                image_idx   = []
                bbox_count  = 0
                for idx, bboxes in enumerate(bboxes_list):
                    for i in range(bboxes.shape[0]):
                        bbox = bboxes[i, :4]
                        output_list.append(img_list[idx][int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
                        image_idx.append(idx)
                        bbox_count += 1
                # for d in output_list:
                #     try:
                #         cv2.resize(d, (0, 0), fx = 1/args.scale, fy = 1/args.scale, interpolation=cv2.INTER_CUBIC)
                #         print('Right shape')
                #         print(0 in d.shape)
                #     except:
                #         print('Wrong shape')
                #         print(0 in d.shape)
                print('Detected count: ', bbox_count)
                output_list = [cv2.resize(d, (0, 0), fx = 1/args.scale, fy = 1/args.scale, interpolation=cv2.INTER_CUBIC) \
                                for d in output_list if not (0 in d.shape)]
                print(f'After filtering too small bbox,\n remain: {len(output_list)}')
                action      = [cv2.imwrite(os.path.join(output_dir, os.path.basename(img_paths[image_idx[i]]))+'_1', d) \
                               if not os.path.exists(os.path.join(output_dir, os.path.basename(img_paths[image_idx[i]]))+'_1') \
                               else  cv2.imwrite(os.path.join(output_dir, os.path.basename(img_paths[image_idx[i]]))+'_2', d)\
                               for i, d in enumerate(output_list)]

        
        else:
            # for the mAP calculation
            # set the prob_thresholds as tiny number to keep track all the bboxs
            prob_thresholds = 0.01
            bboxes_list = create_mtcnn_net_list(img_list, args.mini_lp, device, \
                                                p_model_path='weights/pnet/pnet_best.pth', \
                                                o_model_path='weights/onet/Onet_best.pth', prob_thresholds=prob_thresholds) 
            cls_name  = 'Digits'
            img_paths = [os.path.basename(d) for d in img_paths]
            img_names = [os.path.splitext(d)[0]+'.txt' for d in img_paths]
            for idx, bboxes in enumerate(bboxes_list):
                for i in range(bboxes.shape[0]):
                    bbox = bboxes[i, :4]
                    prob = bboxes[i, -1]
                    with open(os.path.join(output_dir, img_names[idx]), 'a') as f:
                        f.write(f'{cls_name} {prob} {int(bbox[0])} {int(bbox[1])} {int(bbox[2])} {int(bbox[3])}\n')

                if idx % 50 == 0:
                    print(f'{idx} pred files created')

        

def mtcnn_output(img_dir:str=None, output_dir:str=None):
    img_list = mtcnn_visual(img_dir=img_dir,output_dir=output_dir,output=True)


if __name__ == "__main__":
    # eval one pic
    # img_path = "data/20201229/EXT/resize/20201229080446_0EXT.bmp"
    # mtcnn_visual(img_path = img_path)

    # eval a dir
    # dir_path = "data/20201229/EXT/resize"
    # mtcnn_visual(img_dir = dir_path, output_dir='evaluation/MTCNN')

    # output to LPRnet
    dir_path   = "data/20201229/EXT/resize"
    output_dir = "data/20201229/EXT/lpr_data"
    mtcnn_output(img_dir = dir_path, output_dir=output_dir)