#    dl-classification
#    Copyright (C) 2017  Matthieu Ospici
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


import argparse

import numpy as np
import tensorflow as tf
from PIL import Image

import models


class tomato_inf:
    def __init__(self, check_):
        self.img_feed = tf.placeholder(tf.float32)

        self.output_logits = tf.nn.softmax(
            models.foodv_test(
                self.img_feed,
                reg_val=0.0,
                is_train=False,
                dropout_p=1.0))

        self.sess = tf.Session()

        self.checkpoint_name = check_

        saver = tf.train.Saver()
        print("loading model...")

        saver.restore(self.sess, self.checkpoint_name)

        print("Model loaded !")

    def prepare_image(self, image):
        # prepare the image for the neural network
        img = tf.image.resize_images(image, [200, 200])
        img = tf.image.per_image_standardization(img)
        img = tf.reshape(img, [1, 200, 200, 3])

        return img

    def has_tomatoes(self, im_path):
        # load the image
        im = Image.open(im_path)
        im = np.asarray(im, dtype=np.float32)
        im = self.prepare_image(im)

        # launch an inference with the image
        pred = self.sess.run(
            self.output_logits, feed_dict={
                self.img_feed: im.eval(
                    session=self.sess)})

        if np.argmax(pred) == 0:
            print("NOT a tomato ! (confidence : ", pred[0, 0], "%)")
        else:
            print("We have a tomato ! (confidence : ", pred[0, 1], "%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    group_genera = parser.add_argument_group('General options')

    group_genera.add_argument(
        "-img",
        "--img_path",
        help="Path to the image",
        required=True)
    group_genera.add_argument(
        "-cp",
        "--checkpoint",
        help="Path to the checkpoint",
        required=True)

    args = parser.parse_args()

    inf = tomato_inf(args.checkpoint)

    inf.has_tomatoes(args.img_path)
