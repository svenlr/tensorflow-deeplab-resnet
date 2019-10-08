import cv2
import os
from tqdm import tqdm
import joblib


def check_sample(line):
    if '\t' not in line:
        return ''
    jpg_path, seg_img_path = line.split('\t')
    if not os.path.isfile(seg_img_path):
        return ''
    with open(jpg_path, 'rb') as f:
        contents = f.read()
        if len(contents) >= 2:
            check_chars = contents[-2:]
        else:
            print('Empty image: {}'.format(jpg_path))
            return ''
    if check_chars != b'\xff\xd9':
        print('Not complete image: {}'.format(jpg_path))
        return ''
    else:
        return "{}\t{}\n".format(jpg_path, seg_img_path)


def clean_train_list(train_list_contents):
    lines = train_list_contents.split('\n')

    cleaned_list = joblib.Parallel(n_jobs=12)(
        joblib.delayed(check_sample)(line) for line in tqdm(lines, desc="clean data list"))

    output = ''.join(cleaned_list)

    return output
