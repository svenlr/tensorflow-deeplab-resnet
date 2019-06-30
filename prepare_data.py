#!/usr/bin/env python3.6

import sys
import os
import cv2
import numpy as np
from tqdm import tqdm
import joblib

from clean_train_list import clean_train_list


def merge_annotation(annotations_to_merge, segmentations_path, segmentations_combined_path):
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
        joblib.delayed(merge_annotation)(annotations_per_image, segmentations_path, segmentations_combined_path) for
        annotations_per_image in tqdm(image_id_to_annotations_map.values(), desc="combine segmentations"))


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
    return res


def create_segmentation_list_string(base_path):
    images_path = os.path.join(base_path, "images")
    segmentations_path = os.path.join(base_path, "parsing_annos")
    segmentations_combined_path = os.path.join(base_path, "parsing_annos_combined")
    os.makedirs(segmentations_combined_path, exist_ok=True)
    combine_segmentations(segmentations_path, segmentations_combined_path)
    if os.sep in output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    contents = gen_train_list(images_path, segmentations_combined_path)
    contents = clean_train_list(contents)
    return contents


if __name__ == '__main__':
    dataset_path = sys.argv[1]
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')

    output_file = sys.argv[2] if len(sys.argv) > 2 else "train.list"

    contents = create_segmentation_list_string(train_path)
    contents += create_segmentation_list_string(val_path)

    with open(output_file, "w") as f:
        if contents[-1] == '\n':
            contents = contents[:-1]
        f.write(contents)
