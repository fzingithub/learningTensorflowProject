# -*- coding: utf-8 -*-
'''
Created on 2019/5/7
Author: zhe
Email: 1194585271@qq.com
'''
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# w = tf.constant(1.)
# x = tf.constant(2.)
# y = x * w
# with tf.GradientTape() as tape:
#     tape.watch([w])
#     y2 = x * w
#
# grad1 = tape.gradient(y, [w])
# print(grad1)
#
# with tf.GradientTape() as tape:
#     tape.watch([w])
#     y2 = x * w
#
# grad2 = tape.gradient(y2, [w])
# print(grad2)


# persistent

w = tf.constant(1.)
x = tf.constant(2.)
y = x * w
with tf.GradientTape(persistent=True) as tape:
    tape.watch([w])
    y2 = x * w

grad = tape.gradient(y2, [w])
print(grad)
grad = tape.gradient(y2, [w])
print(grad)