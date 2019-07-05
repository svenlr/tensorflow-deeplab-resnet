#!/usr/bin/env python3.6

import sys
import os
import cv2
import numpy as np
from tqdm import tqdm
import joblib

from clean_train_list import clean_train_list

from segmentation_util import *


def merge_annotations(annotation_files_to_merge, segmentations_path, segmentations_combined_path, merger):
    if len(annotation_files_to_merge) == 0:
        return
    annotation_data_list = []
    for ann_file in annotation_files_to_merge:
        src_path = os.path.join(segmentations_path, ann_file)
        annotation_data = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)[:, :, 2]
        if merger.should_drop_whole_image(annotation_data):
            return
        else:
            annotation_data = merger.merge_and_rename_categories(annotation_data)
            if annotation_data is not None:
                annotation_data_list.append(annotation_data)
    identifier = int(annotation_files_to_merge[0].split("_")[0])
    if len(annotation_data_list) > 0:
        dst = np.max(annotation_data_list, axis=0)
        target_path = os.path.join(segmentations_combined_path, "{}.png".format(identifier))
        cv2.imwrite(target_path, dst)
    else:
        print(str(identifier) + " has no annotations")


def combine_segmentations(segmentations_path, seg_combined_path, max_num_images=None):
    os.makedirs(seg_combined_path, exist_ok=True)
    annotation_files = os.listdir(segmentations_path)
    if max_num_images is not None:
        annotation_files = [a for a in annotation_files if 0 <= int(a.split("_")[0]) - max_num_images <= max_num_images]

    merger = SegmentationMerge(MERGE_LISTS_VERY_TINY, DROP_IMAGES_WITH_CATEGORIES_VERY_TINY)

    image_id_to_annotations_map = dict((annotation.split("_")[0], []) for annotation in annotation_files)
    for annotation_file in annotation_files:
        image_id_to_annotations_map[annotation_file.split("_")[0]].append(annotation_file)

    joblib.Parallel(n_jobs=12)(
        joblib.delayed(merge_annotations)(annotations_per_image, segmentations_path, seg_combined_path, merger) for
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


def create_segmentation_list_string(base_path, max_num_images=None):
    images_path = os.path.join(base_path, "images")
    segmentations_path = os.path.join(base_path, "parsing_annos")
    segmentations_combined_path = os.path.join(base_path, "parsing_annos_very_tiny")
    os.makedirs(segmentations_combined_path, exist_ok=True)
    combine_segmentations(segmentations_path, segmentations_combined_path, max_num_images=max_num_images)
    if os.sep in output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    contents_ = gen_train_list(images_path, segmentations_combined_path)
    contents_ = clean_train_list(contents_)
    return contents_


if __name__ == '__main__':
    dataset_path = sys.argv[1]
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')

    output_file = sys.argv[2] if len(sys.argv) > 2 else "train_merge_very_tiny.list"

    max_num_images = int(sys.argv[3]) if len(sys.argv) > 3 else None

    contents = create_segmentation_list_string(train_path, max_num_images=max_num_images)
    contents += create_segmentation_list_string(val_path, max_num_images=max_num_images)

    with open(output_file, "w") as f:
        if contents[-1] == '\n':
            contents = contents[:-1]
        f.write(contents)
