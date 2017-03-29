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
from initializers import weight_variable_xavier,bias_variable2




def foodv_test(images,reg_val = 0.0,is_train=False,dropout_p = 1.0):
  print("6convs, 200x200 input image")
  with tf.variable_scope('conv1') as scope:
    kernel = weight_variable_xavier([5, 5, 3, 48],reg_val)
    bias = bias_variable2([48])
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME') #stride 2
    conv1 = tf.nn.relu(conv + bias, name="conv1")
    #print_activations(conv1)
    

    
    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool1')
    if is_train:
      print("IS TRAIN !")
      pool1 = tf.nn.dropout(pool1, dropout_p)


    print(pool1)

  with tf.variable_scope('conv2') as scope:
    kernel = weight_variable_xavier([5, 5, 48, 64],reg_val)
    bias = bias_variable2([64])
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME') #stride 2
    conv2 = tf.nn.relu(conv + bias, name="conv2")
    #print_activations(conv1)
    

    pool2 = tf.nn.max_pool(conv2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 1,1, 1],
                           padding='SAME',
                           name='pool2')

    if is_train:
      pool2 = tf.nn.dropout(pool2, dropout_p)

  with tf.variable_scope('conv3') as scope:

    kernel = weight_variable_xavier([5, 5, 64, 128],reg_val)
    bias = bias_variable2([128])
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME') 
    conv3 = tf.nn.relu(conv + bias, name="conv3")
    #print_activations(conv1)
    

    pool3 = tf.nn.max_pool(conv3,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool3')

    if is_train:
      pool3 = tf.nn.dropout(pool3, dropout_p)

  with tf.variable_scope('conv4') as scope:
    kernel = weight_variable_xavier([5, 5, 128, 160],reg_val)
    bias = bias_variable2([160])
    conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME') #stride 2
    conv4 = tf.nn.relu(conv + bias, name="conv4")
    #print_activations(conv1)
    

    pool4 = tf.nn.max_pool(conv4,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool4')
    if is_train:
      pool4 = tf.nn.dropout(pool4, dropout_p)



  with tf.variable_scope('conv5') as scope:    
    kernel = weight_variable_xavier([3, 3, 160, 192],reg_val)
    bias = bias_variable2([192])
    conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME') #stride 2
    conv5 = tf.nn.relu(conv + bias, name="conv5")
    #print_activations(conv1)
    

    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool5')

   
    if is_train:
      pool5 = tf.nn.dropout(pool5, dropout_p)


  with tf.variable_scope('conv6') as scope:    
    kernel = weight_variable_xavier([3, 3, 192, 320],reg_val)
    bias = bias_variable2([320])
    conv = tf.nn.conv2d(pool5, kernel, [1, 1, 1, 1], padding='SAME') #stride 2
    conv6 = tf.nn.relu(conv + bias, name="conv6")
    #print_activations(conv1)
    

    pool6 = tf.nn.max_pool(conv6,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool6')

   
    if is_train:
      pool6 = tf.nn.dropout(pool6, dropout_p)

  final_pool = pool6
  print(final_pool)


  size_0 = 7*7*320
  size_last = 4096

  
  with tf.variable_scope('fc1') as scope:
    W_fc1 = weight_variable_xavier([size_0, size_last],reg_val)
    b_fc1 = bias_variable2([size_last])
    h_poolf_flat = tf.reshape(final_pool, [-1, size_0])
    h_fc1 = tf.nn.relu(tf.matmul(h_poolf_flat, W_fc1) + b_fc1)
    #h_fc1 = tf.matmul(h_poolf_flat, W_fc1) + b_fc1
    #_activation_summary(h_fc1)

    if is_train:
      h_fc1 = tf.nn.dropout(h_fc1, dropout_p)            

  with tf.variable_scope('fc2') as scope:
    W_fc2 = weight_variable_xavier([size_last, 2],reg_val)
    b_fc2 = bias_variable2([2])

    h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

 

      
  return h_fc2
  
 


