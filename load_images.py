#    binary-classification
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

import os

import numpy as np
import tensorflow as tf
from scipy import misc


def random_rotate_image(image):
    # rotate image for data augmentation
    angle = np.random.uniform(low=-20.0, high=20.0)
    return misc.imrotate(image, angle, 'bicubic')


def read_labeled_image_list_jpeg(data_paths, ftl):
    # create lists with the filenames and the labels. Classes are balanced

    filenames_path = []

    for dire in data_paths:
        for root, _, files in os.walk(dire, topdown=False):
            filenames_path += [os.path.join(root, name_f) for name_f in files]

    print("Files enumeration done")
    labels = [ftl.to_label(fp) for fp in filenames_path]
    print("Labels generation done", len(filenames_path), len(labels))
    return filenames_path, labels


def create_batch_from_files(
        data_paths,
        l_img_size,
        chan_num,
        batch_size,
        ftl,
        data_aug=False):
    image_list, label_list = read_labeled_image_list_jpeg(data_paths, ftl)

    label_list = tf.convert_to_tensor(label_list, dtype=tf.int32)

    input_queue = tf.train.slice_input_producer([image_list, label_list],
                                                shuffle=True)

    raw_files = input_queue[0]
    image_file = tf.read_file(input_queue[0])
    label = input_queue[1]

    image = tf.image.decode_jpeg(image_file, channels=chan_num)

    if data_aug:  # several operations to increase the dataset
        print("DATA AUGMENTATION ACTIVE")

        image = tf.py_func(random_rotate_image, [image], tf.uint8)
        image = tf.image.random_contrast(image, lower=0.5, upper=0.8)
        image = tf.image.random_brightness(image, 0.4)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

    image = tf.image.resize_images(image, l_img_size)

    image = tf.image.per_image_standardization(image)

    image = tf.reshape(image, [l_img_size[0], l_img_size[1], chan_num])

    # parallel fetch, read_threads should be tunned
    read_threads = 10
    example_list = [(image, label, raw_files) for _ in range(read_threads)]

    batch_size = batch_size
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size

    # batch are randomly created
    image_batch, label_batch, raw_files_batch = tf.train.shuffle_batch_join(
        example_list,
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)

    return image_batch, label_batch, raw_files_batch
