import sys
import numpy as np
import cv2
from deeplab_resnet import decode_labels
from segmentation_util import MERGE_LISTS

if __name__ == '__main__':
    print(len(MERGE_LISTS))
    data_list = sys.argv[1]

    with open(data_list, 'r') as f:
        lines = f.read().split('\n')

    for line in lines:
        jpg_path, seg_path = line.split('\t')

        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)

        seg = np.expand_dims(np.expand_dims(seg, 0), -1)

        msk = decode_labels(seg, num_classes=len(MERGE_LISTS))
        im = msk[0]
        img_o = cv2.imread(jpg_path)

        jpg_path = str(jpg_path)

        img = np.array(im) * 0.9 + np.array(img_o) * 0.7
        img[img > 255] = 255

        cv2.imshow("labels", img_o)
        cv2.waitKey(0)