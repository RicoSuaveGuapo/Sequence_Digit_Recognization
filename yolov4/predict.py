#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def images2video(image_dir_path, output_name, fps):
    img_array = []
    filenames = [os.path.join(image_dir_path,filename) for filename in os.listdir(image_dir_path) if filename.endswith('.jpg')]
    # filenames.sort()
    path = filenames[0]
    img = cv2.imread(path)
    h, w, _ = img.shape
    size = (w, h)
    for filename in filenames:
        img = cv2.imread(filename)
        img_array.append(img)

    out = cv2.VideoWriter(f'/home/rico-li/Job/yolov4-pytorch/predictions/{output_name}.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == "__main__":
    # Prediction part
    yolo = YOLO()
    # path = 'VOCdevkit/VOC2007/JPEGImages'
    # with open('VOCdevkit/VOC2007/ImageSets/Main/test.txt','r') as f:
    #     test_list = f.readlines()
    # test_list = [d.replace('\n','') for d in test_list]

    # images = os.listdir(path)
    # img_path = [os.path.join(path, img) for img in images if os.path.basename(img).split('.')[0] in test_list]

    # if not os.path.exists('predictions/Digits'):
    #     os.mkdir('predictions/Digits')
    # for img in img_path:
    #     image = Image.open(img)
    #     image = image.convert('RGB')
    #     result = yolo.detect_image(image)
    #     result.save(os.path.join('predictions', 'Digits',os.path.basename(img)))
    #     print(f'Prediction on {os.path.basename(img)} is saved')

    # covert result into video
    # fps = 0.7
    # pred_path = 'predictions/Digits'
    # images2video(image_dir_path=pred_path, output_name='Digits_Prediction', fps=fps)


    # VOCdevkit/VOC2007/JPEGImages/20201229175554_0EXT.jpg
    # while True:
    #     img = input('Input image filename:')
    #     try:
    #         image = Image.open(img)
    #         image = image.convert('RGB')
    #     except:
    #         print('Open Error! Try again!')
    #         continue
    #     else:
    #         r_image = yolo.detect_image(image)
    #         r_image.show()



    img = 'VOCdevkit/VOC2007/JPEGImages/20201229175554_0EXT.jpg'
    image = Image.open(img)
    image = image.convert('RGB')

    r_image = yolo.detect_image(image)
    r_image = np.array(r_image)
    plt.imshow(r_image)
    plt.show()