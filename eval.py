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

import tensorflow as tf
from datetime import datetime
import time
import sys
import os
import argparse

import numpy as np 

import models

from load_images import create_batch_from_files
from sklearn import metrics

###### PARSE ARGUMENTS ###
parser = argparse.ArgumentParser()

group_genera = parser.add_argument_group('General options')


group_genera.add_argument("-paths", "--data-path", help="Path to the data to load ",required=True)
group_genera.add_argument("-cp", "--ckpt-path", help="Paths to the checkpoint",required=True)







args = parser.parse_args()
data_path = args.data_path
ckpt_path = args.ckpt_path

print("++++ data path   : ",data_path)
print("++++ ckpt path : ",ckpt_path)

# end arg parsing

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_integer('batch_size',100,"""batch size""")
tf.app.flags.DEFINE_float('dropout',1.0,"""dropout""")



x,label_batch,rfb = create_batch_from_files('doc/technical_test/test_database.txt',data_path,[200,200],3,False)

pred_label = models.foodv_test(x,reg_val=0.0,is_train=False,dropout_p = 1.0)

prediction_label = tf.argmax(pred_label,1)

correct_prediction = tf.equal(tf.cast(tf.argmax(pred_label,1),tf.int32), label_batch)
accuracy_class = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


saver = tf.train.Saver()

init_op = tf.global_variables_initializer()



with tf.Session() as sess:

    sess.run(init_op)

    #create queues to load images
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    
    saver.restore(sess, ckpt_path)

    med_ac = 0
    for step in range(25):
        lb,pb,rf,ac = sess.run([label_batch,prediction_label,rfb,accuracy_class])


        print(rf)
        print(lb)
        print(pb)
        med_ac += ac
        print("Acc : ",ac)
        print("Med : ",med_ac/(step + 1))
        print(metrics.classification_report(lb,pb))


                        
    coord.request_stop()
    coord.join(threads)
