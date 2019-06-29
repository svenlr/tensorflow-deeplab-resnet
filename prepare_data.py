#!/usr/bin/env python3.6

import sys
import os
import cv2
import numpy as np
from tqdm import tqdm
import joblib

from clean_train_list import clean_train_list


def merge_annotation(annotations_to_merge):
    if len(annotations_to_merge) == 0:
        return
    images = []
    for ann_file in annotations_to_merge:
        src_path = os.path.join(segmentations_path, ann_file)
        images.append(cv2.imread(src_path, cv2.IMREAD_UNCHANGED)[:, :, 2])
    dst = np.max(images, axis=0)
    identifier = int(annotations_to_merge[0].split("_")[0])
    target_path = os.path.join(segmentations_combined_path, "{}.png".format(identifier))
    cv2.imwrite(target_path, dst)


def combine_segmentations(segmentations_path, segmentations_combined_path):
    os.makedirs(segmentations_combined_path, exist_ok=True)
    annotation_files = os.listdir(segmentations_path)

    image_id_to_annotations_map = dict((annotation.split("_")[0], []) for annotation in annotation_files)
    for annotation_file in annotation_files:
        image_id_to_annotations_map[annotation_file.split("_")[0]].append(annotation_file)

    joblib.Parallel(n_jobs=12)(
        joblib.delayed(merge_annotation)(annotations_per_image) for annotations_per_image in
        tqdm(image_id_to_annotations_map.values(), desc="combine segmentations"))


def gen_train_list(images_directory, segmentations_directory):
    images_files = os.listdir(images_directory)
    res = ""
    progress_bar = tqdm(total=len(images_files))
    for f_img in images_files:
        progress_bar.update(1)
        f_seg = f_img.replace(".jpg", ".png")
        seg_path = os.path.join(segmentations_directory, f_seg)
        if os.path.isfile(seg_path):
            img_path = os.path.join(images_directory, f_img)
            res += img_path + "\t" + seg_path + "\n"
    return res[:-1]


if __name__ == '__main__':
    train_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "train.list"
    images_path = os.path.join(train_path, "images")
    segmentations_path = os.path.join(train_path, "parsing_annos")
    segmentations_combined_path = os.path.join(train_path, "parsing_annos_combined")
    segmentations_combined_1c_path = os.path.join(train_path, "parsing_annos_combined_1c")
    os.makedirs(segmentations_combined_path, exist_ok=True)
    os.makedirs(segmentations_combined_1c_path, exist_ok=True)
    combine_segmentations(segmentations_path, segmentations_combined_path)
    if os.sep in output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    contents = gen_train_list(images_path, segmentations_combined_path)
    contents = clean_train_list(contents)

    with open(output_file, "w") as f:
        f.write(contents)
