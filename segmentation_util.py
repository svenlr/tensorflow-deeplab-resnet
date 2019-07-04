from typing import Optional

import numpy as np

CATEGORIES = ["background", "cap/hat", "helmet", "face", "hair", "left-arm", "right-arm", "left-hand", "right-hand",
              "protector", "bikini/bra", "jacket/windbreaker/hoodie", "t-shirt", "polo-shirt", "sweater", "singlet",
              "torso-skin", "pants", "shorts/swim-shorts", "skirt", "stockings", "socks", "left-boot", "right-boot",
              "left-shoe", "right-shoe", "left-highheel", "right-highheel", "left-sandal", "right-sandal", "left-leg",
              "right-leg", "left-foot", "right-foot", "coat", "dress", "robe", "jumpsuit", "other-full-body-clothes",
              "headwear", "backpack", "ball", "bats", "belt", "bottle", "carrybag", "cases", "sunglasses", "eyewear",
              "glove", "scarf", "umbrella", "wallet/purse", "watch", "wristband", "tie", "other-accessary",
              "other-upper-body-clothes", "other-lower-body-clothes"]

CATEGORY_TO_IDX = dict((name, i) for i, name in enumerate(CATEGORIES))

MERGE_LISTS = ["background",
               ["cap/hat", "helmet", "headwear"],
               "face", "hair",
               ["left-arm", "right-arm", "watch", "wristband"],
               ["left-hand", "right-hand", "glove"],
               "bikini/bra", "jacket/windbreaker/hoodie", "t-shirt", "polo-shirt", "sweater", "singlet",
               "torso-skin", "pants", "shorts/swim-shorts", "skirt", "stockings", "socks",
               ["left-boot", "right-boot", "left-shoe", "right-shoe", "left-sandal", "right-sandal"],
               ["left-highheel", "right-highheel"],
               ["left-leg", "right-leg"],
               ["left-foot", "right-foot"],
               "coat",
               ["dress", "ball"],
               "jumpsuit", "other-full-body-clothes", "backpack", "ball", "belt", "tie", "carrybag", "cases",
               "sunglasses", "eyewear", "scarf", "wallet/purse", "other-upper-body-clothes", "other-lower-body-clothes"]

MERGE_LISTS_DENSE = [
    "background",
    ["cap/hat", "helmet", "headwear"],
    "face", "hair",
    ["left-arm", "right-arm", "watch", "wristband"],
    ["left-hand", "right-hand", "glove"],
    "bikini/bra",
    ["jacket/windbreaker/hoodie", "coat"],
    "sweater",
    ["t-shirt", "polo-shirt", "singlet"],
    "torso-skin", "pants", "shorts/swim-shorts", "skirt",
    ["left-boot", "right-boot", "left-shoe", "right-shoe", "left-sandal", "right-sandal"],
    ["left-highheel", "right-highheel"],
    ["left-leg", "right-leg"],
    ["left-foot", "right-foot"],
    ["socks", "stockings"],
    ["dress", "ball"],
    "jumpsuit",
    "belt",
    "tie",
    "scarf",
    "sunglasses",
    "eyewear",
    "backpack",
    "cases",
    ["wallet/purse", "carrybag"]]

MERGE_LISTS_EXTREME = [
    "background",
    ["cap/hat", "helmet", "headwear"],
    "face", "hair",
    ["left-arm", "right-arm", "watch", "wristband"],
    ["left-hand", "right-hand", "glove"],
    "bikini/bra",
    ["jacket/windbreaker/hoodie", "coat", "sweater", "t-shirt", "polo-shirt", "singlet", "other-upper-body-clothes"],
    "torso-skin",
    ["pants", "shorts/swim-shorts", "skirt", "other-lower-body-clothes"],
    ["left-boot", "right-boot", "left-shoe", "right-shoe", "left-sandal", "right-sandal", "left-highheel",
     "right-highheel", "left-foot", "right-foot"],
    ["left-leg", "right-leg", "socks", "stockings"],
    ["dress", "jumpsuit", "other-full-body-clothes"],
    "belt",
    "tie",
    "scarf",
    "sunglasses",
    "eyewear",
    ["backpack", "cases", "wallet/purse", "carrybag"]]

MERGE_LISTS_TINY = [
    "background",
    ["cap/hat", "helmet", "headwear"],
    "face",
    "hair",
    ["left-leg", "right-leg"],
    ["left-arm", "right-arm", "watch", "wristband"],
    ["left-hand", "right-hand", "glove"],
    ["jacket/windbreaker/hoodie", "coat", "sweater", "t-shirt", "polo-shirt", "singlet", "other-upper-body-clothes",
     "pants", "shorts/swim-shorts", "skirt", "other-lower-body-clothes", "dress", "jumpsuit", "other-full-body-clothes",
     "tie", "scarf", "bikini/bra", "stockings", "belt", "robe"],
    "torso-skin",
    ["left-boot", "right-boot", "left-shoe", "right-shoe", "left-sandal", "right-sandal", "left-highheel",
     "right-highheel", "left-foot", "right-foot", "socks"],
    "sunglasses",
    "eyewear"]

DROP_IMAGES_WITH_CATEGORIES = ["robe"]

DROP_IMAGES_WITH_CATEGORIES_DENSE = ["robe", "other-full-body-clothes", "other-upper-body-clothes",
                                     "other-lower-body-clothes"]

DROP_IMAGES_WITH_CATEGORIES_EXTREME = []

DROP_IMAGES_WITH_CATEGORIES_TINY = []


class SegmentationMerge:
    def __init__(self, merge_lists, drop_images_with_categories):
        self.drop_images_with_categories = drop_images_with_categories
        self.dropped_category_ids = [CATEGORY_TO_IDX[category] for category in self.drop_images_with_categories]
        self.change_id_map = dict(
            (CATEGORY_TO_IDX[name], i) for i, name in enumerate(merge_lists) if isinstance(name, str))

        for i in range(len(CATEGORIES)):
            name = CATEGORIES[i]
            if name in merge_lists:
                new_idx = merge_lists.index(name)
                self.change_id_map[i] = new_idx
            else:
                for j, merge_list in enumerate(merge_lists):
                    if name in merge_list:
                        self.change_id_map[i] = j
                        break

    def merge_and_rename_categories(self, label):
        # type: (np.ndarray) -> Optional[np.ndarray]
        out_label = np.zeros(label.shape)
        for old_idx, new_idx in self.change_id_map.items():
            out_label[label == old_idx] = new_idx
        if np.all(out_label == 0):
            return None
        else:
            return out_label

    def should_drop_whole_image(self, label):
        class_num_idx = np.argmax(label > 0)
        class_num = np.ravel(label)[class_num_idx]
        if class_num in self.dropped_category_ids:
            return True
        else:
            return False
