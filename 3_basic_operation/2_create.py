import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# from numpy or list

a = np.ones([3,1])
print (a)
aa = tf.convert_to_tensor(a)
print (aa)

bb = tf.convert_to_tensor([1,1,1])
print (bb)

print (tf.constant(1))
print (tf.constant([1]))
print (tf.constant([1, 2.]))


# tf.zeros

print (tf.zeros([]))
print (tf.zeros([1]))
print (tf.zeros([1,2]))

# tf.zeros_like
print (tf.zeros_like(bb))

# tf.ones
print (tf.zeros([]))
print (tf.zeros([1]))
print (tf.zeros([1,2]))

# tf.ones_like

print (tf.ones_like(bb))


# tf.fill
print (tf.fill([2,3],4))

# tf.random.normal

print (tf.random.normal([2, 2], mean=0, stddev=1))


# tf.random.truncated_normal
print (tf.random.truncated_normal([2,2], mean=0, stddev=1, dtype=tf.double))

# tf.random.uniform
print (tf.random.uniform([2, 2], minval=0, maxval=1))


# Random Permutation
idx = tf.range(200)
idx = tf.random.shuffle(idx)
print (idx)

a =  tf.random.truncated_normal([200, 784])
b = tf.random.uniform([200], maxval=10, dtype=tf.int32)
print (a)
print (b)

a = tf.gather(a, idx)
b = tf.gather(b, idx)
print (a)
print (b)