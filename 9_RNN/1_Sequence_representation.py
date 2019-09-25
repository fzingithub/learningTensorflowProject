# -*- coding: utf-8 -*-
'''
Created on 2019/6/4
Author: zhe
Email: 1194585271@qq.com
'''


import tensorflow as tf
from tensorflow.keras import layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.range(5)
x = tf.random.shuffle(x)
print(x)


net  = layers.Embedding(10, 4)
print(net(x))

print(net.trainable)

print(net.trainable_variables)