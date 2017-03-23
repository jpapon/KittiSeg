# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:50:47 2015

@author: teichman
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import numpy as np
import scipy as scp
import scipy.misc
import sys
from random import shuffle
import zipfile
from glob import glob

import logging
reload(logging)

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

IMAGE_FOLDER = "depth_image"
GT_FOLDER = "gt_image"
DATA_DIR = "../DATA/data_RCTA/carry"
TRAIN_FILE_OUT = "train.txt"
VAL_FILE_OUT = "val.txt"
VAL_PROP = 0.05

def make_val_split(file_tuples, data_folder):
    """
    Splits the Images in train and test.
    Assumes a File all.txt in data_folder.
    """


    test_num = len (file_tupes) * VAL_PROP

    shuffle(file_tuples)

    train = file_tuples[:-test_num]
    test = file_tuples[-test_num:]

    train_file = os.path.join(data_folder, TRAIN_FILE_OUT)
    test_file = os.path.join(data_folder, VAL_FILE_OUT)

    with open(train_file, 'w') as file:
        for label in train:
            file.write(label[0] +' ' + label[1]+'\n')

    with open(test_file, 'w') as file:
        for label in test:
            file.write(label[0] +' ' + label[1]+'\n')

def get_file_tuples (data_dir):
    image_folder = os.path.join(data_dir, 'training',IMAGE_FOLDER)
    gt_folder = os.path.join(data_dir, 'training',GT_FOLDER)

    image_files = glob(os.path.join(image_folder,'*.png'))
    image_files = [ os.path.join('training',IMAGE_FOLDER,os.path.basename(image_file)) for image_file in image_files]
    gt_image_files = glob(os.path.join(gt_folder,'*.png'))
    gt_image_file_dic = {os.path.basename(gt_file) : os.path.join('training',GT_FOLDER,os.path.basename(gt_file)) for gt_file in gt_image_files}
    file_tuples = []
    for image_file in image_files:
      basename = os.path.basename(image_file)
      if (basename in gt_image_file_dic):
        file_tuples.append((image_file, gt_image_file_dic[basename]))

    print ("Found {} matching file pairs".format(len(file_tuples)))
    return file_tuples

def main():


    #zip_file = "data_road.zip"
    #zip_file = os.path.join(data_dir, zip_file)
    #if not os.path.exists(zip_file):
    #    logging.error("File not found: %s", zip_file)
    #    exit(1)
    #zipfile.ZipFile(zip_file, 'r').extractall(data_dir)
    file_tuples = get_file_tuples (DATA_DIR)
    make_val_split(file_tuples, DATA_DIR)


if __name__ == '__main__':
    main()
