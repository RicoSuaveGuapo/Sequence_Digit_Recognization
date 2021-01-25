import cv2 as cv
import cv2
import os
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

data_path = 'data/20201229'


def images2video(image_dir_path, output_name, des_path, fps=1):
    img_array = []
    filenames = os.listdir(image_dir_path)
    filenames = [os.path.join(image_dir_path,filename) for filename in filenames if (filename.endswith('.bmp')) or (filename.endswith('.png'))]
    for filename in filenames:
        img = cv2.imread(filename)
        h, w, _ = img.shape
        img_array.append(img)
    
    out = cv2.VideoWriter(f'{os.path.join(des_path,output_name)}', cv2.VideoWriter_fourcc(*"MJPG"), fps, (h, w))
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print(f'Video output to {output_name}')


def build_scheduler(optimizer, name):
    if name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode = 'min', patience=6)
    elif name == 'StepLR':
        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

    return scheduler


# image normalized
def normalize_img(df):
    img_paths = df['file_name'].tolist()
    img_paths = [os.path.join(data_path, d) for d in img_paths]
    
    img       = [cv.imread(d,0) for d in img_paths]
    mean_list = [np.mean(d) for d in img]
    mean      = np.sum(mean_list)/len(mean_list)
    return mean

def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
        or list of numpy.array if boxes.size(0) > 1
    """
    # box = (x1, y1, x2, y2)
    n_bbox = boxes.shape[0]
    if n_bbox == 2:
        box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        area = []
        ovr  = []
        for i in range(n_bbox):
            area.append((boxes[i, 2] - boxes[i, 0] + 1) * (boxes[i, 3] - boxes[i, 1] + 1))
            # obtain the offset of the interception of union between crop_box and gt_box
            xx1 = np.maximum(box[0], boxes[i, 0])
            yy1 = np.maximum(box[1], boxes[i, 1])
            xx2 = np.minimum(box[2], boxes[i, 2])
            yy2 = np.minimum(box[3], boxes[i, 3])
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h
            ovr.append(inter / (box_area + area[i] - inter))
        
    elif n_bbox == 4:
        box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        area     = (boxes[2] - boxes[0] + 1) * (boxes[3] - boxes[1] + 1)

        # obtain the offset of the interception of union between crop_box and gt_box
        xx1 = np.maximum(box[0], boxes[0])
        yy1 = np.maximum(box[1], boxes[1])
        xx2 = np.minimum(box[2], boxes[2])
        yy2 = np.minimum(box[3], boxes[3])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        ovr  = inter / (box_area + area - inter)

    elif n_bbox == 1:
        box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        area     = (boxes[0,2] - boxes[0,0] + 1) * (boxes[0,3] - boxes[0,1] + 1)

        # obtain the offset of the interception of union between crop_box and gt_box
        xx1 = np.maximum(box[0], boxes[0,0])
        yy1 = np.maximum(box[1], boxes[0,1])
        xx2 = np.minimum(box[2], boxes[0,2])
        yy2 = np.minimum(box[3], boxes[0,3])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        ovr  = inter / (box_area + area - inter)
    return ovr

def correct_bboxes(bboxes, width, height):
    """Crop boxes that are too big and get coordinates
    with respect to cutouts.

    Arguments:
        bboxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        width: a float number.
        height: a float number.

    Returns:
        dy, dx, edy, edx: a int numpy arrays of shape [n],
            coordinates of the boxes with respect to the cutouts.
        y, x, ey, ex: a int numpy arrays of shape [n],
            corrected ymin, xmin, ymax, xmax.
        h, w: a int numpy arrays of shape [n],
            just heights and widths of boxes.

        in the following order:
            [dy, edy, dx, edx, y, ey, x, ex, w, h].
    """

    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    x2,y2 = np.clip(x2, x1, None), np.clip(y2, y1, None)
    w, h = x2 - x1 + 1.0,  y2 - y1 + 1.0
    num_boxes = bboxes.shape[0]

    # 'e' stands for end
    # (x, y) -> (ex, ey)
    x, y, ex, ey = x1, y1, x2, y2

    # we need to cut out a box from the image.
    # (x, y, ex, ey) are corrected coordinates of the box
    # in the image.
    # (dx, dy, edx, edy) are coordinates of the box in the cutout
    # from the image.
    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0

    # if box's bottom right corner is too far right
    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0

    # if box's bottom right corner is too low
    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0

    # if box's top left corner is too far left
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0

    # if box's top left corner is too high
    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype('int32') for i in return_list]

    return return_list


def calibrate_box(bboxes, offsets):
    """Transform bounding boxes to be more like true bounding boxes.
    'offsets' is one of the outputs of the nets.

    Arguments:
        bboxes: a float numpy array of shape [n, 5].
        offsets: a float numpy array of shape [n, 4].

    Returns:
        a float numpy array of shape [n, 5].
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)

    # this is what happening here:
    # tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
    # x1_true = x1 + tx1*w
    # y1_true = y1 + ty1*h
    # x2_true = x2 + tx2*w
    # y2_true = y2 + ty2*h
    # below is just more compact form of this

    # are offsets always such that
    # x1 < x2 and y1 < y2 ?

    translation = np.hstack([w, h, w, h])*offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes

def nms(boxes, overlap_threshold=0.5, mode='union'):
    """Non-maximum suppression.

    Arguments:
        boxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        overlap_threshold: a float number.
        mode: 'union' or 'min'.

    Returns:
        list with indices of the selected boxes
    """

    # if there are no boxes, return the empty list
    if len(boxes) == 0:
        return []

    # list of picked indices
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1.0)*(y2 - y1 + 1.0)
    ids = np.argsort(score)  # in increasing order

    while len(ids) > 0:

        # grab index of the largest value
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        # compute intersections
        # of the box with the largest score
        # with the rest of boxes

        # left top corner of intersection boxes
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])

        # right bottom corner of intersection boxes
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])

        # width and height of intersection boxes
        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)

        # intersections' areas
        inter = w * h
        if mode == 'min':
            overlap = inter/np.minimum(area[i], area[ids[:last]])
        elif mode == 'union':
            # intersection over union (IoU)
            overlap = inter/(area[i] + area[ids[:last]] - inter)

        # delete all boxes where overlap is too big
        ids = np.delete(
            ids,
            np.concatenate([[last], np.where(overlap > overlap_threshold)[0]])
        )

    return pick


def draw_anno(df, which, des_path:dir):
    '''
    To show the annotation of images
    parameters
        which: 
            0: the upper bbox to show
            1: the lower bbox to show
            2: show both
    '''
    
    if not os.path.exists(des_path):
        os.mkdir(des_path)
    img_paths   = df['file_name'].tolist()
    img_paths   = [os.path.join(data_path, 'EXT','resize', d) for d in img_paths]
    imgs        = [cv.imread(img_path,0) for img_path in img_paths]

    if which == 0:
        gt   = df['GT_1'].tolist()
        x1y1 = list(zip(df['xmin'].tolist(), df['ymin'].tolist()))
        x2y2 = list(zip(df['xmax'].tolist(), df['ymax_2'].tolist()))
    elif which == 1:
        gt   = df['GT_2'].tolist()
        x1y1 = list(zip(df['xmin'].tolist(), df['ymax_2'].tolist()))
        x2y2 = list(zip(df['xmax'].tolist(), df['ymax'].tolist()))
        text_ymax = df['ymax'].tolist()
        text_ymax = [d+40 for d in text_ymax]
        text = list(zip(df['xmin'].tolist(), text_ymax))
    else:
        gt_1   = df['GT_1'].tolist()
        gt_2   = df['GT_2'].tolist()
        x1y1_1 = list(zip(df['xmin'].tolist(), df['ymin'].tolist()))
        x2y2_1 = list(zip(df['xmax'].tolist(), df['ymax_2'].tolist()))
        x1y1_2 = list(zip(df['xmin'].tolist(), df['ymax_2'].tolist()))
        x2y2_2 = list(zip(df['xmax'].tolist(), df['ymax'].tolist()))
        text_ymax = df['ymax'].tolist()
        text_ymax = [d+40 for d in text_ymax]
        text = list(zip(df['xmin'].tolist(), text_ymax))
        
    color     = (255, 0, 0)
    thickness = 2
    fontScale = 1.5
    font       = cv.FONT_HERSHEY_SIMPLEX 
    if which in [0,1]:
        img_rec = [cv.rectangle(img, x1y1[idk], x2y2[idk], color, thickness)  for idk, img in enumerate(imgs)]
        if which == 0:
            img_rec = [cv.putText(img, str(gt[idk]), x1y1[idk], font, fontScale, color, thickness)\
                    for idk, img in enumerate(imgs)]
        else:
            img_rec = [cv.putText(img, str(gt[idk]), text[idk], font, fontScale, color, thickness)\
                    for idk, img in enumerate(imgs)]
        action  = [cv.imwrite(os.path.join(des_path,os.path.basename(img_paths[idk])), \
                    img) for idk, img in enumerate(img_rec)]
        print('Example:')
        plt.imshow(img_rec[0])
        plt.show()
    else:
        img_rec = [cv.rectangle(img, x1y1_1[idk], x2y2_1[idk], color, thickness)  for idk, img in enumerate(imgs)]
        img_rec = [cv.rectangle(img, x1y1_2[idk], x2y2_2[idk], color, thickness)  for idk, img in enumerate(img_rec)]
        img_rec = [cv.putText(img, str(gt_1[idk]), x1y1_1[idk], font, fontScale, color, thickness)\
             for idk, img in enumerate(imgs)]
        img_rec = [cv.putText(img, str(gt_2[idk]), text[idk], font, fontScale, color, thickness)\
             for idk, img in enumerate(imgs)]
        action  = [cv.imwrite(os.path.join(des_path,os.path.basename(img_paths[idk])), \
                    img) for idk, img in enumerate(img_rec)]
        print('Example:')
        plt.imshow(img_rec[0])
        plt.show()
    print(f'Output annotated files to {des_path}')

if __name__ == "__main__":
    images2video('tmp_result/LPRnet_result/', 'LPRnet_upper.avi', 'tmp_result/LPRnet_result/', 1)