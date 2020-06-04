"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px.

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
from absl import flags
import numpy as np
import pickle

import skimage.io as io
import tensorflow as tf

from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

flags.DEFINE_string('img_path', 'data', 'Directory of images to run')
flags.DEFINE_string('json_path', 'openpose', 'If specified, uses the openpose output to crop the image.')
flags.DEFINE_string('pkl_path', 'data', 'output pkl')


def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    if scale is None:
        return None, None, None

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def main(img_path, pkl_path, json_path):
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    count = len(os.listdir(json_path))
    for ix, f in enumerate(os.listdir(json_path)):
        if not f.endswith(".json"):
            continue

        print('Processing %d/%d: %s' % (ix+1, count, f))

        js = os.path.join(json_path, f)

        fname = f.replace('_keypoints.json', '.jpg')
        img = os.path.join(img_path, fname)

        input_img, proc_param, img = preprocess_image(img, js)
        if input_img is None:
            print('no keypoints')
            continue

        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(input_img, 0)
        joints, verts, cams, joints3d, theta = model.predict(input_img, get_theta=True)

        out_name = f.replace('_keypoints.json', '_mesh.pkl')
        out_path = os.path.join(pkl_path, out_name)

        with open(out_path, 'wb') as outf:
            pickle.dump([joints, verts, cams, joints3d, theta], outf)


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL
    config.batch_size = 1

    main(config.img_path, config.pkl_path, config.json_path)
