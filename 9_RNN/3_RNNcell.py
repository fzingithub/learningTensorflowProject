# -*- coding: utf-8 -*-
'''
Created on 2019/6/21
Author: zhe
Email: 1194585271@qq.com
'''

import os
import tensorflow as tf
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# input dim and hidden dim
cell = layers.SimpleRNNCell(3)   #simple differance LSTM and GRU
cell.build(input_shape=(None, 4))
# h = x_t@w_xh+h_t-1@w_hh + bias

print(cell.trainable_variables)


# Single layer RNN Cell
x = tf.random.normal([4, 80, 100])
xt0  = x[:,0,:]

cell = layers.SimpleRNNCell(64)   #simple differance LSTM and GRU

out, xt1 = cell(xt0, [tf.zeros([4, 64])])


print(out.shape, xt1[0].shape)

print(id(out), id(xt1[0]))

print(cell.trainable_variables)


# Multi-Layers RNN

x = tf.random.normal([4, 80, 100])
xt0  = x[:,0,:]
print(xt0.shape)

cell1 = layers.SimpleRNNCell(64)
cell2 = layers.SimpleRNNCell(64)

state0 = [tf.zeros([4,64])] # memery0
state1 = [tf.zeros([4,64])] # memery1


out0, state0 = cell1(xt0, state0)
out1, state1 = cell2(out0, state1)

print(out1.shape, state1[0].shape)
print(id(out1), id(state1[0]))