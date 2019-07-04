import sys
import numpy as np
import cv2
import deeplab_resnet.utils
from deeplab_resnet import decode_labels
from segmentation_util import *

if __name__ == '__main__':
    data_list = sys.argv[1]

    with open(data_list, 'r') as f:
        lines = f.read().split('\n')

    print(len(MERGE_LISTS_TINY))

    i = 0
    for line in lines:
        jpg_path, seg_path = line.split('\t')

        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)

        seg = np.expand_dims(np.expand_dims(seg, 0), -1)

        # class_idx = MERGE_LISTS.index("carrybag") - 2
        # class_idx = 18
        # if np.all(seg != class_idx):
        #     continue
        # print(i)
        # i+= 1
        # continue
        # seg[seg != class_idx] = 0

        msk = decode_labels(seg, num_classes=len(MERGE_LISTS))
        im = msk[0]
        img_o = cv2.imread(jpg_path)

        jpg_path = str(jpg_path)

        print(im.shape, im.dtype)
        print(img_o.shape, img_o.dtype)
        # img = np.array(im) * 0.9 + np.array(img_o) * 0.7
        img = np.hstack([im, img_o])
        img[img > 255] = 255

        cv2.imshow("labels", img.astype(np.uint8))
        cv2.waitKey(0)