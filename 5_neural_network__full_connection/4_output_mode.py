# -*- coding: utf-8 -*-
'''
Created on 2019/5/6
Author: zhe
Email: 1194585271@qq.com
'''
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# sigmoid

a = tf.linspace(-6., 6, 10)
print(tf.reduce_min(a), tf.reduce_max(a))
a = tf.sigmoid(a)
print(tf.reduce_min(a), tf.reduce_max(a))


a = tf.random.normal([1, 28*28])*5
print(tf.reduce_min(a), tf.reduce_max(a))

a = tf.sigmoid(a)
print(tf.reduce_min(a), tf.reduce_max(a))




a = tf.linspace(-2., 2, 5)
print(tf.reduce_sum(tf.sigmoid(a)))

print(tf.reduce_sum(tf.nn.softmax(a)))



# softmax

logits = tf.random.uniform([1, 10], minval=-2, maxval=2)
print(logits)
# tf.Tensor(
# [[ 0.39318037 -1.1405406  -1.4796648  -1.059619    0.00410414  0.21543264
#    0.9332652  -1.734467   -0.23943186  1.796689  ]], shape=(1, 10), dtype=float32)

prob = tf.nn.softmax(logits, axis=1)
print(prob)
# tf.Tensor(
# [[0.02795015 0.0291838  0.08102272 0.01900888 0.0333832  0.07500502
#   0.23073502 0.07967106 0.31747448 0.10656565]], shape=(1, 10), dtype=float32)

print(tf.reduce_sum(prob))   # tf.Tensor(1.0, shape=(), dtype=float32)


# tanh
a = tf.linspace(-6., 6, 10)
print(tf.reduce_min(a), tf.reduce_max(a))
a = tf.tanh(a)
print(tf.reduce_min(a), tf.reduce_max(a))