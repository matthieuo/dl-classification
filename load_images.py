import json

import tensorflow as tf

import os
import numpy as np
from scipy import misc

FLAGS = tf.app.flags.FLAGS


def random_rotate_image(image):
    #rotate image for data augmentation
    angle = np.random.uniform(low=-20.0, high=20.0)
    return misc.imrotate(image, angle, 'bicubic')


def read_labeled_image_list_jpeg(file_json,prefix):
    #create lists with the filenames and the labels. Classes are balanced

    with open(file_json) as data_file:
        data = json.load(data_file)


    l_f = []
    l_oth = []

    for d in data:
        _, file_extension = os.path.splitext(os.path.join(prefix,d["name"]))

        if file_extension == ".png":continue

        find_t = False
        for bo in  d["boxes"]:
            if bo["id"] == "939030726152341c154ba28629341da6_train": #tomato id

                find_t = True
        if find_t:

            for _ in range(12):  #to balance the training /test set
                l_f.append(os.path.join(prefix,d["name"]))
            
        else:
            l_oth.append(os.path.join(prefix,d["name"]))


        


    filenames_path = l_f
    labels = [1]*len(l_f)
    filenames_path += l_oth
    labels += [0]*len(l_oth)
   
    # we let tensorflow randomly picking the batches from the lists
    
    return  filenames_path, labels



def create_batch_from_files(json_f,prefix,l_img_size,chan_num,data_aug = False):
    image_list, label_list = read_labeled_image_list_jpeg(json_f,prefix)

    label_list = tf.convert_to_tensor(label_list, dtype=tf.int32)

    input_queue = tf.train.slice_input_producer([image_list, label_list],
                                                shuffle=True)

    raw_files = input_queue[0]
    image_file = tf.read_file(input_queue[0])
    label = input_queue[1]
    
    image = tf.image.decode_jpeg(image_file,channels=chan_num)


    if data_aug: #several operations to increase the dataset
        print("DATA AUGMENTATION ACTIVE")

        image = tf.py_func(random_rotate_image, [image], tf.uint8)
        image = tf.image.random_contrast(image, lower=0.5, upper=0.8)
        image = tf.image.random_brightness(image, 0.4)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)


    image = tf.image.resize_images(image,l_img_size)

    image = tf.image.per_image_standardization(image)

    image = tf.reshape(image,[l_img_size[0],l_img_size[1],chan_num])

    #parallel fetch, read_threads should be tunned
    read_threads = 10
    example_list = [(image,label,raw_files)  for _ in range(read_threads)]

    batch_size=FLAGS.batch_size
    min_after_dequeue = 15000
    capacity = min_after_dequeue + 3 * batch_size

    #batch are randomly created
    image_batch,label_batch,raw_files_batch = tf.train.shuffle_batch_join(
        example_list,
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)



    return image_batch,label_batch,raw_files_batch
