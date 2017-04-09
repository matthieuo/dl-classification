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
import numpy as np


def bias_variable2(shape):
  return tf.get_variable("biais2", shape,initializer=tf.constant_initializer(0.1))


# -- Xavier init with get_variable
def weight_variable_xavier(shape,reg_val = 0.0):
  W_xav = tf.get_variable("weight_xavier", shape,
                          initializer = tf.contrib.layers.xavier_initializer(),
                          regularizer = tf.contrib.layers.l2_regularizer(reg_val))
  
  return W_xav


  





