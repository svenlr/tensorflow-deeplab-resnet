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
               ["left-arm", "right-arm"],
               ["left-hand", "right-hand"],
               "bikini/bra", "jacket/windbreaker/hoodie", "t-shirt", "polo-shirt", "sweater", "singlet",
               "torso-skin", "pants", "shorts/swim-shorts", "skirt", "stockings", "socks",
               ["left-boot", "right-boot", "left-shoe", "right-shoe", "left-sandal", "right-sandal"],
               ["left-highheel", "right-highheel"],
               ["left-leg", "right-leg"],
               ["left-foot", "right-foot"],
               "coat",
               ["dress", "ball"],
               "jumpsuit", "other-full-body-clothes", "backpack", "ball", "belt", "carrybag", "cases", "sunglasses",
               "eyewear", "scarf", "wallet/purse", "other-upper-body-clothes", "other-lower-body-clothes"]

DROP_IMAGES_WITH_CATEGORIES = ["robe"]

CHANGE_ID_MAP  = dict((CATEGORY_TO_IDX[name], i) for i, name in enumerate(MERGE_LISTS) if isinstance(name, str))

for i in range(len(CATEGORIES)):
    name = CATEGORIES[i]
    if name in MERGE_LISTS:
        new_idx = MERGE_LISTS.index(name)
        CHANGE_ID_MAP[i] = new_idx
    else:
        for j, merge_list in enumerate(MERGE_LISTS):
            if name in merge_list:
                CHANGE_ID_MAP[i] = j
                break


def merge_and_rename_categories(label):
    # type: (np.ndarray) -> Optional[np.ndarray]
    out_label = np.zeros(label.shape)
    for old_idx, new_idx in CHANGE_ID_MAP.items():
        out_label[label == old_idx] = new_idx
    if np.all(out_label == 0):
        return None
    else:
        return out_label
