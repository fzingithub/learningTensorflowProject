import tensorflow as tf
import numpy as np

# TF is a computing lib
'''
int, float, double
bool
string
'''
tf.constant(1)
tf.constant(1.)

# tf.constant(2.2, dtype=tf.int32)   # error
tf.constant(2, dtype=tf.double)

tf.constant([True, False])

tf.constant('Hello world!')



# Tensor Property

with tf.device('cpu'):
    a = tf.constant([1])

with tf.device('gpu:0'):
    b = tf.constant([2])

print (a.device)
print (b.device)

a.ndim
a.shape
aa = a.gpu()
aa.device
b.numpy

print (tf.rank(tf.constant([1,2,3,4,5])))
print (tf.ones([728, 512]).shape)




# Check Tensor Type

# convert

# bool2int

# tf.Variable

# 2numpy

