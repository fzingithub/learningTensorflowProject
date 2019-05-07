# -*- coding: utf-8 -*-
'''
Created on 2019/5/6
Author: zhe
Email: 1194585271@qq.com
'''
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

w = tf.Variable(1.0)
b = tf.Variable(2.0)
x = tf.Variable(3.0)

with tf.GradientTape() as t1:
  with tf.GradientTape() as t2:
    y = x * w + b
  dy_dw, dy_db = t2.gradient(y, [w, b])
d2y_dw2 = t1.gradient(dy_dw, w)

print(dy_dw)   # tf.Tensor(3.0, shape=(), dtype=float32)
print(dy_db)   # tf.Tensor(1.0, shape=(), dtype=float32)
print(d2y_dw2)   # None

assert dy_dw.numpy() == 3.0
assert d2y_dw2 is None