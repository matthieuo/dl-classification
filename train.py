#    Tomatoes classifier
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


group_genera.add_argument("-path", "--data-path", help="Path to the data to load",required=True)

group_train = parser.add_argument_group('Train control')



group_train.add_argument("-dp", "--dropout", help="Dropout value, 1 = no dropout",type=float,required=True)
group_train.add_argument("-reg", "--reg-fact", help="Regularization",type=float,required=True)
group_train.add_argument("-bp", "--base_path", help="Base path for log",required=True)

args = parser.parse_args()

data_path = args.data_path

if(args.dropout == 1):
    dp_str = "No"
    dropout_f = False
else:
    dp_str = str(args.dropout)
    dropout_f = True


output_dir = "_dp_" + dp_str + "_rf_"+str(args.reg_fact)

output_path = os.path.join(args.base_path,output_dir)


print(output_path)

print("++++ data path   : ",data_path)
print("---- output path : ",output_path)


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', output_path,
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 50000000,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_integer('batch_size',64,"""batch size""")
tf.app.flags.DEFINE_float('dropout',args.dropout,"""dropout""")


reg_fact = args.reg_fact

## END PARSE ARGS ####


print("Configuration resume")

print("reg fact = ",reg_fact)
print("dropout = ", dp_str)
print("output = ",output_path)

train_val = True

with tf.device('/cpu:0'): #data augmentation on CPU to increase perf
    x,label_batch,_ = create_batch_from_files('doc/technical_test/train_database.txt',data_path,[200,200],3,True)

pred_label = models.foodv_test(x,reg_val=reg_fact,is_train=train_val,dropout_p = args.dropout)



if tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
    reg_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
else:
    reg_losses = 0.0


cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_label,
                                                               labels=label_batch,
                                                               name='cross_entropy')


#global loss with the cross entropy and the L2 reg
loss =  reg_losses + cross_entropy


correct_prediction = tf.equal(tf.cast(tf.argmax(pred_label,1),tf.int32), label_batch)
accuracy_class = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


prediction_label = tf.argmax(pred_label,1)


tf.summary.scalar("acc_class", accuracy_class)

train_operations = tf.train.AdamOptimizer(1e-4).minimize(loss)

# Create a saver.
saver = tf.train.Saver(tf.global_variables())



summary_op = tf.summary.merge_all() # for tensorboard



init_op = tf.global_variables_initializer()





with tf.Session() as sess:

    sess.run(init_op)



    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)



    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)


    for step in range(FLAGS.max_steps):
            start_time = time.time()
 
            __ = sess.run(train_operations)
        
        
            duration = time.time() - start_time
      
        
            if step % 100 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, -1,
                                     examples_per_sec, sec_per_batch))
            
                print("wrote sumary op")
               
            
                sum_op,ac,lb,pb = sess.run([summary_op,accuracy_class,label_batch,prediction_label])
                

                print("ac ",ac)
                print(lb)
                print(pb)

                print(metrics.classification_report(lb,pb))


                summary_writer.add_summary(sum_op, step)

                
            if step % 1000 == 0 and  step > 0: #save a checkpoint every 1000 iterations
                print("sav checkpoint")
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                print("done")

                        
    coord.request_stop()
    coord.join(threads)
