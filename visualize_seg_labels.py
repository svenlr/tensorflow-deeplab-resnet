import argparse
import os
import sys
import numpy as np
import cv2

from deeplab_resnet import decode_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare segmentation data")
    parser.add_argument("data_list", type=str,
                        help="list with (image, label) pairs")
    parser.add_argument("data_dir", type=str, default="",
                        help="Path to the directory of the dataset.")
    args = parser.parse_args()

    with open(args.data_list, 'r') as f:
        lines = f.read().split('\n')

    print(len(lines))

    i = 0
    for line in lines:
        img_path, seg_path = line.split('\t')
        img_path = os.path.join(args.data_dir, img_path)
        seg_path = os.path.join(args.data_dir, seg_path)

        if not os.path.exists(seg_path):
            print("skip non-existent: "+ seg_path)
            continue
        if not os.path.exists(img_path):
            print("skip non-existent: "+ img_path)
            continue

        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)

        seg = np.expand_dims(np.expand_dims(seg, 0), -1)

        msk = decode_labels(seg, num_classes=np.max(seg) + 1)
        im = msk[0]
        img_o = cv2.imread(img_path)

        img_path = str(img_path)

        print(im.shape, im.dtype)
        print(img_o.shape, img_o.dtype)
        # img = np.array(im) * 0.9 + np.array(img_o) * 0.7
        img = np.hstack([im, img_o])
        img[img > 255] = 255

        cv2.imshow("labels", img.astype(np.uint8))
        cv2.waitKey(0)
