#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Marvin Teichmann


"""
Detects Cars in an image using KittiSeg.

Input: Image
Output: Image (with Cars plotted in Green)

Utilizes: Trained KittiSeg weights. If no logdir is given,
pretrained weights will be downloaded and used.

Usage:
python demo.py --input_image data/demo.png [--output_image output_image]
                [--logdir /path/to/weights] [--gpus 0]


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

import collections

# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf

from matplotlib import cm
import matplotlib as mpl

from scipy.ndimage.filters import sobel

flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')

from seg_utils import seg_utils as seg

try:
    # Check whether setup was done correctly

    import tensorvision.utils as tv_utils
    import tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)


flags.DEFINE_string('logdir', None,
                    'Path to logdir.')
flags.DEFINE_string('input_image', None,
                    'Input image name')
flags.DEFINE_string('color_image', None,
                    'color image')

flags.DEFINE_string('output_image', None,
                    'Output image name')
flags.DEFINE_string('outdir', None,
                    'Output directory')

default_run = 'KittiSeg_pretrained'
weights_url = ("ftp://mi.eng.cam.ac.uk/"
               "pub/mttt2/models/KittiSeg_pretrained.zip")

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

def maybe_download_and_extract(runs_dir):
    logdir = os.path.join(runs_dir, default_run)

    if os.path.exists(logdir):
        # weights are downloaded. Nothing to do
        return

    import zipfile
    download_name = tv_utils.download(weights_url, runs_dir)

    logging.info("Extracting KittiSeg_pretrained.zip")

    zipfile.ZipFile(download_name, 'r').extractall(runs_dir)

    return


def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image

def main(_):
    tv_utils.set_gpus_to_use()

    if FLAGS.input_image is None:
        logging.error("No input_image was given.")
        logging.info(
            "Usage: python run_classifier.py --input_image data/test.png "
            "[--output_image output_image] [--logdir /path/to/weights] "
            "[--gpus GPUs_to_use] ")
        exit(1)

    if FLAGS.logdir is None:
        # Download and use weights from the MultiNet Paper
        if 'TV_DIR_RUNS' in os.environ:
            runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                    'KittiSeg')
        else:
            runs_dir = 'RUNS'
        maybe_download_and_extract(runs_dir)
        logdir = os.path.join(runs_dir, default_run)
    else:
        logging.info("Using weights found in {}".format(FLAGS.logdir))
        logdir = FLAGS.logdir
    if FLAGS.outdir is None:
      outdir = '.'
    else:
      outdir = FLAGS.outdir
      if not os.path.exists(outdir):
        os.makedirs(outdir)

    if FLAGS.color_image is None:
        color_image = None
    else:
        color_image = scp.misc.imread(FLAGS.color_image).astype(np.float)/255.0
        
    # Loading hyperparameters from logdir
    hypes = tv_utils.load_hypes_from_logdir(logdir)

    logging.info("Hypes loaded successfully.")

    # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
    modules = tv_utils.load_modules_from_logdir(logdir)
    logging.info("Modules loaded successfully. Starting to build tf graph.")

    # Create tf graph and build module.
    with tf.Graph().as_default():
        # Create placeholder for input
        image_pl = tf.placeholder(tf.float32)
        image = tf.expand_dims(image_pl, 0)

        # build Tensorflow graph using the model from logdir
        prediction = core.build_inference_graph(hypes, modules,
                                                image=image)

        logging.info("Graph build successfully.")

        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()

        # Load weights from logdir
        core.load_weights(logdir, sess, saver)

        logging.info("Weights loaded successfully.")

    input_image = FLAGS.input_image
    logging.info("Starting inference using {} as input".format(input_image))

    # Load and resize input image
    image_in = scp.misc.imread(input_image, mode = 'F')
    image,gt_image = preprocess_depth_image (image_in)
    if (len(image.shape) == 2):
      image = np.expand_dims (image, axis = 2)

    if hypes['jitter']['reseize_image']:
        # Resize input only, if specified in hypes
        image_height = hypes['jitter']['image_height']
        image_width = hypes['jitter']['image_width']
        image = scp.misc.imresize(image, size=(image_height, image_width),
                                  interp='cubic')

    # Run KittiSeg model on image
    feed = {image_pl: image}
    softmax = prediction['softmax']
    output = sess.run([softmax], feed_dict=feed)

    # Reshape output from flat vector to 2D Image
    shape = image.shape
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
      norm = mpl.colors.Normalize(vmin=0.5, vmax=3.5)
      m = cm.ScalarMappable(norm=norm, cmap=cmap)
      gray_image = m.to_rgba(image_in)[:,:,0:3]
    if (color_image is None):
        color_image = gray_image
        
    output = output[0].reshape(shape[0], shape[1], num_classes)
    label_im = np.argmax(output, axis = 2)
    label_overlay_im = tv_utils.overlay_segmentation(color_image, label_im, label_colors_alpha)
    label_im = tv_utils.overlay_segmentation(np.zeros_like(color_image), label_im, label_colors)

    # Plot graspability prob as red-blue overlay
    graspability_class = 2
    grasp_prob_image = output[:,:,graspability_class]
    grasp_rb_image = seg.make_overlay(color_image, grasp_prob_image)

    # Plot ungraspable prob as red-blue overlay
    ungraspability_class = 1
    ungrasp_prob_image = output[:,:,ungraspability_class]
    ungrasp_rb_image = seg.make_overlay(color_image, ungrasp_prob_image)


    # Save output images to disk.
    if FLAGS.output_image is None:
        output_base_name = input_image
    else:
        output_base_name = FLAGS.output_image
    
    print ("*"*25)
    print (output_base_name)
    print ("*"*25)

    basename = os.path.splitext(os.path.basename(output_base_name))[0]
    scaled_image_name = os.path.join(outdir, basename + '_scaled.png')
    grasp_rb_image_name = os.path.join(outdir,basename + '_grasp_prob.png')
    ungrasp_rb_image_name = os.path.join(outdir,basename + '_ungrasp_prob.png')
    label_overlay_image_name = os.path.join(outdir,basename + '_label_overlay.png')
    label_image_name = os.path.join(outdir,basename + '_labels.png')
    
    scp.misc.imsave(scaled_image_name, np.squeeze(image))
    scp.misc.imsave(grasp_rb_image_name, grasp_rb_image)
    scp.misc.imsave(ungrasp_rb_image_name, ungrasp_rb_image)
    scp.misc.imsave(label_overlay_image_name, label_overlay_im)
    scp.misc.imsave(label_image_name, label_im)
    
    logging.info("")
    logging.info("scaled input image has been saved to: {}".format(scaled_image_name))
    logging.info("Red-Blue overlay of graspability prob have been saved to: {}".format(grasp_rb_image_name))
    logging.info("Red-Blue overlay of ungraspability prob have been saved to: {}".format(ungrasp_rb_image_name))
    logging.info("Label plot of predictions have been saved to: {}".format(label_image_name))


if __name__ == '__main__':
    tf.app.run()
