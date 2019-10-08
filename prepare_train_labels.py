#!/usr/bin/env python3

import argparse
import os
import sys

import cv2
import joblib
from tqdm import tqdm

from labels_util import SegmentationMerge, parse_train_list, parse_categories_file, data_pairs_to_train_list


def _merge(pair, merger, merged_labels_directory, data_dir, sub_directory):
    img_path, seg_path = pair
    dst_seg_path = os.path.join(merged_labels_directory, os.path.basename(seg_path))
    seg = cv2.imread(os.path.join(data_dir, sub_directory, seg_path), cv2.IMREAD_GRAYSCALE)
    out_seg = merger.merge_and_rename_categories(seg)
    cv2.imwrite(dst_seg_path, out_seg)
    return os.path.join(sub_directory, img_path), dst_seg_path.replace(data_dir, "").strip(os.sep)


def main(args):
    categories = None
    if not args.sub_dirs or len(args.sub_dirs) == 0:
        tmp_sub_directories = os.listdir(args.data_dir)
        sub_dirs = []
        for d in tmp_sub_directories:
            if "train.list" in os.listdir(os.path.join(args.data_dir, d)):
                sub_dirs.append(d)
                print("Auto detected training data directory: " + os.path.join(args.data_dir, d))
    else:
        sub_dirs = args.sub_dirs
    if len(sub_dirs) == 0:
        sub_dirs = [""]
    if args.categories_file:
        categories = parse_categories_file(args.categories_file)
    if categories is None:
        for sub_directory in args.sub_dirs:
            path_to_categories_file = os.path.join(args.data_dir, sub_directory, "categories")
            if os.path.exists(path_to_categories_file):
                categories = parse_categories_file(path_to_categories_file)
                print("Auto detected categories file: " + path_to_categories_file)
    if categories is None:
        print("Error: no category list given and auto detect failed.")
        print("Data dir: " + args.data_dir)
        print("Subdirectories: " + "  ".join(sub_dirs))
        sys.exit(1)
    drop_categories = []
    if args.drop_categories:
        drop_categories = args.drop_categories
    if args.categories_merge_file:
        with open(args.categories_merge_file) as f:
            lines = f.read().split("\n")
        merge_list = [c.split(" ") for c in lines if c.strip() != ""]
        for i in range(len(merge_list)):
            merge_list[i] = [c.strip() for c in merge_list[i] if c.strip() != ""]
        merge_list = [(m if len(m) > 1 else m[0]) for m in merge_list if len(m) > 0]
    else:
        merge_list = categories

    identifier = str(len(merge_list)) + "_raw" + str(len(categories))
    if args.categories_merge_file:
        identifier += "_" + str(args.categories_merge_file).split(os.sep)[-1]
    identifier += "_" + "_".join(sub_dirs)

    merger = SegmentationMerge(categories, merge_list, drop_images_with_categories=drop_categories)

    final_pairs = []
    for sub_directory in sub_dirs:
        sub_list_file = os.path.join(args.data_dir, sub_directory, "train.list")
        merged_labels_directory = os.path.join(args.data_dir, sub_directory, "labels_" + identifier)
        os.makedirs(merged_labels_directory, exist_ok=True)
        pairs = parse_train_list(sub_list_file)

        tmp_final_pairs = joblib.Parallel(n_jobs=6)(
            joblib.delayed(_merge)(pair, merger, merged_labels_directory, args.data_dir, sub_directory) for pair in tqdm(pairs, desc="process labels"))
        final_pairs += tmp_final_pairs

    with open(os.path.join(args.output_directory, "train_" + identifier + ".list"), "w+") as f:
        f.write(data_pairs_to_train_list(final_pairs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare segmentation data")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Path to the parent directory of the datset.")
    parser.add_argument("--sub-dirs", nargs="+", type=str,
                        help="Subfolders of the parent directory to be included in the train.list file. Each must have a train.list file on its own.")
    parser.add_argument("--output-directory", type=str, default=".",
                        help="where to write the output train.list file")
    parser.add_argument("--categories-file", type=str, required=False,
                        help="Lists the category names of the label files in order (each line a category)")
    parser.add_argument("--categories-merge-file", type=str, required=False,
                        help="A file that defines the index of categories and which categories are merged into one.")
    parser.add_argument("--drop-categories", nargs="+", type=str, default=[], required=False,
                        help="which categories to drop")

    main(parser.parse_args())
