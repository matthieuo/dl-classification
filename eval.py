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

import tensorflow as tf
from datetime import datetime
import time
import sys
import os
import argparse

import numpy as np 

import models
import utils

from load_images import create_batch_from_files
from sklearn import metrics





def test_model(test_path,
               num_classes,
               log_path,
               ftl):
    
    with tf.device('/cpu:0'):
        x,label_batch,rfb = create_batch_from_files(test_path,[200,200],3,100,ftl,False)

    pred_label = models.foodv_test(x,num_classes,reg_val=0.0,is_train=False,dropout_p = 1.0)



    assert num_classes == ftl.curr_class,"Number of classes found on datasets are not equal to the number of classes given"
        
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

        last_chk = tf.train.latest_checkpoint(log_path)
    
        chk_step = last_chk.split("-")[-1]
        print(chk_step)
        saver.restore(sess, last_chk)
     

        med_ac = 0
        for step in range(25):
            lb,pb,rf,ac = sess.run([label_batch,prediction_label,rfb,accuracy_class])


            #print(rf)
            print(lb)
            print(pb)
            med_ac += ac
            print("Acc : ",ac)
            print("Med : ",med_ac/(step + 1))
            print(metrics.classification_report(lb,pb))


            
        coord.request_stop()
        coord.join(threads)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    group_genera = parser.add_argument_group('General options')


    group_genera.add_argument("-paths", "--data-paths", help="Path to the test set",required=True)
   
    group_genera.add_argument("-ti", "--test_identification", help="String to identify test for tensorboard",required=True)

    group_genera.add_argument("-lp", "--log_path", help="Log directory path",required=True)

    group_genera.add_argument("-nc", "--num-classes", help="Number of classes on the training set", type=int,required=True)

    args = parser.parse_args()


    data_path = args.data_paths.split(';')
    
    print("++++ data path   : ",data_path)
    print("++++ log path : "   ,args.log_path)


    ftl = utils.file_to_label_binary()

    
    test_model(data_path,
               args.num_classes,
               args.log_path,
               ftl)
