#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the model network using a queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy as scp
import random
from seg_utils import seg_utils as seg
from scipy.ndimage.filters import sobel

import tensorflow as tf
import time

import tensorvision
import tensorvision.utils as tv_utils
import copy
import matplotlib as mpl
from matplotlib import cm as cm

def preprocess_depth_image (image,gt_image = None):

    MAX_DELTA = 64
    MIN_DELTA = -64
    MIN_GRAD_MAG = 2/255.0
    sbx = sobel(image, axis = 1).astype(float)
    sby = sobel(image, axis = 0).astype(float)

    sbx = np.clip(sbx,MIN_DELTA, MAX_DELTA)
    sby = np.clip(sby,MIN_DELTA, MAX_DELTA)

    sbx = (sbx  - MIN_DELTA)
    sby = (sby  - MIN_DELTA)

    sb = np.hypot(sbx,sby)
    max_val = 2 * (MAX_DELTA**2 + MIN_DELTA**2)**0.5
    sb = np.clip(sb,0,max_val) / max_val

    sbx = sbx / (MAX_DELTA - MIN_DELTA)
    sby = sby / (MAX_DELTA - MIN_DELTA)

    sb_stack = np.dstack ((sb, sbx, sby))

    if (gt_image is not None):
      #Mask out values with zero gradient in GT
      not_valid_mask = sb_stack[:,:,0] < MIN_GRAD_MAG
      gt_image[not_valid_mask] = np.zeros(gt_image.shape)[not_valid_mask]

    return sb_stack, gt_image
    '''

    MAX_DEPTH = 2000
    MIN_DEPTH = 400
    image = np.clip (image, MIN_DEPTH, MAX_DEPTH)
    image = (image - MIN_DEPTH).astype(np.float32)
    #Adjust to 0-1
    image = image / (MAX_DEPTH-MIN_DEPTH)

    #Mask out clipped values in GT
    if (gt_image is not None):
      not_valid_mask = np.logical_or (image <= 0, image >= 1.0)
      gt_image[not_valid_mask] = np.ones(gt_image.shape)[not_valid_mask]

    #Now make 0-255 and uint8
    #image = (image * 255).astype(np.uint8)
    return image, gt_image
    '''
def eval_image(hypes, valid_gt, class_gt, cnn_image):
    """."""
    thresh = np.array(range(0, 256))/255.0

    FN, FP, posNum, negNum = seg.evalExp(class_gt, cnn_image,
                                         thresh, validMap=None,
                                         validArea=valid_gt)

    return FN, FP, posNum, negNum


def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image


def evaluate(hypes, sess, image_pl, inf_out):

    softmax = inf_out['softmax']
    data_dir = hypes['dirs']['data_dir']
    data_file = hypes['data']['val_file']
    data_file = os.path.join(data_dir, data_file)
    image_dir = os.path.dirname(data_file)

    thresh = np.array(range(0, 256))/255.0
    total_fp = np.zeros(thresh.shape)
    total_fn = np.zeros(thresh.shape)
    total_posnum = 0
    total_negnum = 0

    image_list = []

    label_colors = {}
    for key in hypes['data'].keys():
      if ('_color' in key):
        color = np.array(hypes['data'][key])
        label_colors[color[0]] = (color[1],color[2],color[3],128)

    with open(data_file) as file:
        for i, datum in enumerate(file):
                datum = datum.rstrip()
                image_file, gt_file = datum.split(" ")
                image_file = os.path.join(image_dir, image_file)
                gt_file = os.path.join(image_dir, gt_file)

                image = scp.misc.imread(image_file, mode='F')
                image_in = copy.deepcopy(image)
                gt_image = scp.misc.imread(gt_file)
                #Expand dims, and process to fit range in 0-255
                image,gt_image = preprocess_depth_image (image,gt_image)
                if (len(image.shape) == 2):
                  image = np.expand_dims (image, axis = 2)

                if hypes['jitter']['fix_shape']:
                    shape = image.shape
                    image_height = hypes['jitter']['image_height']
                    image_width = hypes['jitter']['image_width']
                    assert(image_height >= shape[0])
                    assert(image_width >= shape[1])

                    offset_x = (image_height - shape[0])//2
                    offset_y = (image_width - shape[1])//2
                    new_image = np.zeros([image_height, image_width, 3])
                    new_image[offset_x:offset_x+shape[0],
                              offset_y:offset_y+shape[1]] = image
                    input_image = new_image
                elif hypes['jitter']['reseize_image']:
                    image_height = hypes['jitter']['image_height']
                    image_width = hypes['jitter']['image_width']
                    gt_image_old = gt_image
                    image, gt_image = resize_label_image(image, gt_image,
                                                         image_height,
                                                         image_width)
                    input_image = image
                else:
                    input_image = image

                shape = input_image.shape

                feed_dict = {image_pl: input_image}

                output = sess.run([softmax], feed_dict=feed_dict)

                num_classes = hypes['arch']['num_classes']
                class_of_interest = 2
                channel_of_interest = 1
                class_score_im = output[0].reshape(shape[0], shape[1], num_classes)
                class_score_im = class_score_im[:,:,class_of_interest]

                if hypes['jitter']['fix_shape']:
                    gt_shape = gt_image.shape
                    class_score_im = class_score_im[offset_x:offset_x+gt_shape[0],
                                          offset_y:offset_y+gt_shape[1]]

                if True:

                    label_colors = {}
                    label_colors_alpha = {}
                    for key in hypes['data'].keys():
                      if ('_color' in key):
                        color = np.array(hypes['data'][key])
                        label_colors[color[0]] = (color[1],color[2],color[3],255)
                        label_colors_alpha[color[0]] = (color[1],color[2],color[3],128)
                    num_classes = hypes['arch']['num_classes']

                    label_colors_alpha[0] = (0,0,0,128)
                    label_colors_alpha[2] = (0,255,0,128)
                    label_colors_alpha[1] = (255,0,0,128)

                    label_colors[0] = (0,0,0,255)
                    label_colors[2] = (0,255,0,255)
                    label_colors[1] = (255,0,0,255)

                    if (len(image_in.shape) == 2):
                      cmap = cm.gray
                      norm = mpl.colors.Normalize(vmin=0.5, vmax=2.0)
                      m = cm.ScalarMappable(norm=norm, cmap=cmap)
                      gray_image = m.to_rgba(image_in)[:,:,0:3]
                    else:
                      gray_image = image_in

                    if (image.shape[2] == 1):
                      color_image = np.repeat(image, 3, axis = 2)
                    else:
                      color_image = image

                    output = output[0].reshape(shape[0], shape[1], num_classes)
                    label_im = np.argmax(output, axis = 2)
                    label_overlay_im = tv_utils.overlay_segmentation(color_image, label_im, label_colors_alpha)
                    label_im = tv_utils.overlay_segmentation(np.zeros_like(gray_image), label_im, label_colors)
                    name = os.path.basename(image_file)

                    name2 = name.split('.')[0] + '_labels.png'
                    image_list.append((name2, label_overlay_im))

                    # Plot graspability prob as red-blue overlay
                    graspability_class = 2
                    grasp_prob_image = output[:,:,graspability_class]
                    grasp_rb_image = seg.make_overlay(gray_image, grasp_prob_image)


                valid_gt = np.logical_or (gt_image[:, :, 0] > 1, gt_image[:, :, 1] > 1)
                valid_gt = np.logical_or (valid_gt, gt_image[:, :, 2] > 1)


                class_gt = gt_image[:,:,channel_of_interest] > 0
                FN, FP, posNum, negNum = eval_image(hypes, valid_gt, class_gt, class_score_im)

                total_fp += FP
                total_fn += FN
                total_posnum += posNum
                total_negnum += negNum

    eval_dict = seg.pxEval_maximizeFMeasure(total_posnum, total_negnum,
                                            total_fn, total_fp,
                                            thresh=thresh)

    start_time = time.time()
    for i in range(10):
        sess.run([softmax], feed_dict=feed_dict)
    dt = (time.time() - start_time)/10

    eval_list = []

    eval_list.append(('MaxF1', 100*eval_dict['MaxF']))
    eval_list.append(('BestThresh', 100*eval_dict['BestThresh']))
    eval_list.append(('Average Precision', 100*eval_dict['AvgPrec']))
    eval_list.append(('Speed (msec)', 1000*dt))
    eval_list.append(('Speed (fps)', 1/dt))

    return eval_list, image_list
