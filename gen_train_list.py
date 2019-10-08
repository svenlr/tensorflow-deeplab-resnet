#!/usr/bin/env python

import argparse
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare segmentation data")
    parser.add_argument("mode", type=str, choices=["gazebo_plugin"],
                        help="annotation format type")
    parser.add_argument("data_path", type=str,
                        help="Path to the directory of the dataset.")
    args = parser.parse_args()

    pairs = []
    if args.mode == "gazebo_plugin":
        files = os.listdir(args.data_path)
        label_files = [f for f in files if "label" in f]
        for label_file in label_files:
            img_file = label_file.replace("_labels", "")
            pairs.append((img_file, label_file))

    content = ""
    for pair in pairs:
        content += pair[0] + "\t" + pair[1] + "\n"
    content = content[:-1]  # cut last \n

    with open(os.path.join(args.data_path, "train.list"), "w+") as f:
        f.write(content)
