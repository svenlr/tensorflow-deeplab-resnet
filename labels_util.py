import os
import numpy as np
from typing import Optional
from tqdm import tqdm


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


def parse_train_list(path):
    with open(path) as f:
        path_contents = f.read().split("\n")
        path_contents = [l.strip() for l in path_contents if l.strip() != ""]
    pairs = []
    for line in path_contents:
        if "\t" in line:
            img_path, seg_path = line.split("\t")
        else:
            while "  " in line:
                line = line.replace("  ", " ")
            img_path, seg_path = line.split(" ")
        pairs.append((img_path, seg_path))
    return pairs


def data_pairs_to_train_list(pairs):
    content = ""
    for pair in pairs:
        content += pair[0] + "\t" + pair[1] + "\n"
    content = content[:-1]
    return content


def parse_categories_file(path):
    with open(path) as f:
        categories_ = f.read().split("\n")
    categories_ = [c.strip() for c in categories_ if c.strip() != ""]
    return categories_


class SegmentationMerge:
    def __init__(self, categories, category_merge_list, drop_images_with_categories=None):
        self.drop_images_with_categories = drop_images_with_categories if drop_images_with_categories is not None else []
        category_to_idx_map = dict((name, i) for i, name in enumerate(categories))
        self.dropped_category_ids = [category_to_idx_map[category] for category in self.drop_images_with_categories]
        self.change_id_map = dict(
            (category_to_idx_map[name], i) for i, name in enumerate(category_merge_list) if isinstance(name, str))

        for i in range(len(categories)):
            name = categories[i]
            if name in category_merge_list:
                new_idx = category_merge_list.index(name)
                self.change_id_map[i] = new_idx
            else:
                for j, merge_list in enumerate(category_merge_list):
                    if name in merge_list:
                        self.change_id_map[i] = j
                        break

        new_id_map = dict((categories[i], j) for i,j in self.change_id_map.items())
        print(new_id_map)

    def merge_and_rename_categories(self, label):
        # type: (np.ndarray) -> Optional[np.ndarray]
        out_label = np.zeros(label.shape, dtype=label.dtype)
        for old_idx, new_idx in self.change_id_map.items():
            out_label[label == old_idx] = new_idx
        return out_label

    def should_drop_whole_image(self, label):
        class_num_idx = np.argmax(label > 0)
        class_num = np.ravel(label)[class_num_idx]
        if class_num in self.dropped_category_ids:
            return True
        else:
            return False
