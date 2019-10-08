"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#import cv2
from pathlib import Path

from PIL import Image

import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label

import pdb
import h5py

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

NUM_CLASSES = 59
DATA_LIST = '/media/isf/15eea210-d66d-451a-a423-736787aecdd3/isf-loewen/dll_data/images/data_list_ny.txt'
SAVE_DIR = './output/'


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("img_path", type=str,
                        help="Path to the RGB image file folder.")
    parser.add_argument("model_weights", type=str,
                        help="Path to the file with model weights.")
    parser.add_argument("--data_list", type=str, default=DATA_LIST,
                        help="Path to the image list.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    return parser.parse_args()


def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def file_len(fname):
    with open(fname) as f:
        contents = f.read()
    return len(contents.split('\n'))


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    # remove_huge_images(args.data_list, args.img_path)
    num_steps = file_len(args.data_list)
    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    print(args.img_path, ' ', file_len(args.data_list))
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.img_path,
            args.data_list,
            None,  # No defined input size.
            False,  # No random scale.
            False,  # No random mirror.
            255,
            IMG_MEAN,
            coord)
        image, label = reader.image, reader.label
        title = reader.queue[0]
    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0)  # Add one batch dimension.

    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes, is_inference=True)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    fc1_voc12_layer = net.layers['fc1_voc12_stock']
    raw_output_up = tf.image.resize_bilinear(fc1_voc12_layer, tf.shape(image_batch)[1:3, ])
    # uncomment to see only stock segmentation
    # raw_output_up = tf.slice(raw_output_up, [0,0,0,0], [-1,-1,-1,7])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Predictions 2.
    fc1_voc12_extra = net.layers['fc1_voc12_extra']
    raw_output_up_extra = tf.image.resize_bilinear(fc1_voc12_extra, tf.shape(image_batch)[1:3, ])
    # uncomment to see only stock segmentation
    # raw_output_up = tf.slice(raw_output_up, [0,0,0,0], [-1,-1,-1,7])
    raw_output_up_extra = tf.argmax(raw_output_up_extra, dimension=3)
    pred_extra = tf.expand_dims(raw_output_up_extra, dim=3)

    # Set up TF session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, args.model_weights)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    start_time = time.time()
    os.makedirs(args.save_dir, exist_ok=True)

    path_parts = args.img_path.split("/")
    if path_parts[-1].strip() == "":
        path_parts = path_parts[:-1]
    if path_parts[0] == "":
        path_parts[0] = "/"
    bottleneck_dir = os.path.join(*path_parts[:-1], path_parts[-1] + "_hp_bottlenecks")
    os.makedirs(bottleneck_dir, exist_ok=True)

    # Perform inference.
    for step in range(num_steps):
        jpg_name = None
        try:
            preds, preds_extra, jpg_path, fc1_voc12_val = sess.run([pred, pred_extra, title, fc1_voc12_layer])

            msk = decode_labels(preds_extra, num_classes=args.num_classes)
            im = Image.fromarray(msk[0])
            img_o = Image.open(jpg_path)

            jpg_path = str(jpg_path)

            jpg_name = Path(jpg_path).name.split('.')[0]
            img = np.array(im) * 0.9 + np.array(img_o) * 0.7
            img[img > 255] = 255
            img = Image.fromarray(np.uint8(img))
            img.save(os.path.join(args.save_dir, str(jpg_name + '_extra.png')))


            msk = decode_labels(preds, num_classes=args.num_classes)
            im = Image.fromarray(msk[0])
            img_o = Image.open(jpg_path)

            jpg_path = str(jpg_path)

            jpg_name = Path(jpg_path).name.split('.')[0]
            img = np.array(im) * 0.9 + np.array(img_o) * 0.7
            img[img > 255] = 255
            img = Image.fromarray(np.uint8(img))
            img.save(os.path.join(args.save_dir, str(jpg_name + '.png')))

            bottleneck_path = os.path.join(bottleneck_dir, jpg_name + "_hp_bottleneck.h5")
            with h5py.File(bottleneck_path, "w") as bottleneck_file:
                bottleneck_file.create_dataset("fc1_voc12", data=fc1_voc12_val)
            print('Image processed {}.png'.format(jpg_name))
            print('Wrote human parsing bottleneck to {}'.format(bottleneck_path))
        except Exception as e:
            print(e)
            print('Image failed: ', jpg_name)

    total_time = time.time() - start_time
    print('The output files have been saved to {}'.format(args.save_dir))
    print('It took {} sec on each image.'.format(total_time / num_steps))


def remove_huge_images(path_to_data_list, img_path):
    image_file_names = open(path_to_data_list, 'r').readlines()
    new_lines = []
    counter = 0
    for image_file_name in image_file_names:
        path_to_file = os.path.join(img_path, image_file_name).strip()
        image = cv2.imread(path_to_file)
        img_size = 1
        # print(image_file_name)
        # print(np.array(image).shape)
        for dimension in np.array(image).shape:
            img_size *= dimension
        if img_size > 1300 * 1300 * 3:
            counter += 1
            continue
        else:
            new_lines.append(image_file_name)

    with open(path_to_data_list, 'w') as f:
        for line in new_lines:
            f.write(line)

    print('removed ' + str(counter) + ' images because of their size')


if __name__ == '__main__':
    # with open('/media/isf/15eea210-d66d-451a-a423-736787aecdd3/isf-loewen/dll_data/images/data_list_ny.txt', 'r') as f:
    #     print(f.readlines())
    main()
