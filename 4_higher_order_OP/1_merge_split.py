import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# concat

a = tf.ones([4, 35, 8])
b = tf.ones([2, 35, 8])

c = tf.concat([a, b], axis=0)
print (c.shape)   #(6, 35, 8)


a = tf.ones([4, 35, 8])
b = tf.ones([4, 3, 8])

c = tf.concat([a, b], axis=1)
print (c.shape)   # (4, 38, 8)


# stack
a = tf.ones([4, 35, 8])
b = tf.ones([4, 35, 8])

c = tf.concat([a, b], axis=-1)
print (c.shape)   # (4, 35, 16)
d = tf.stack([a, b])
print (d.shape)   # (2, 4, 35, 8)
e = tf.stack([a, b], axis=3)
print (e.shape)   # (4, 35, 8, 2)


# dim mismatch
a = tf.ones([4,35,8])
b = tf.ones([3,33,8])

# c = tf.concat([a, b], axis=0)   # mismatch
b = tf.ones([2, 35, 8])
c = tf.concat([a, b], axis=0)
print (c.shape)   # (6, 35, 8)


# c = tf.stack([a, b])   # mismatch

# unstack

a = tf.ones([4, 35, 8])
b = tf.ones([4, 35, 8])
c = tf.stack([a, b])
print (c.shape)   # (2, 4, 35, 8)

aa, bb = tf.unstack(c, axis=0)
print (aa.shape, bb.shape)   # (4, 35, 8) (4, 35, 8)

res = tf.unstack(c, axis=3)  # shuffle
print (res[0].shape, res[7].shape)   # (2, 4, 35) (2, 4, 35)


# split  vs.  unstack
res = tf.unstack(c, axis=3)  # shuffle
print (len(res))   # 8

res = tf.split(c, axis=3, num_or_size_splits=2)
print (len(res))   # 2

res = tf.split(c, axis=3, num_or_size_splits=[3,3,2])
print (len(res), res[0].shape, res[1].shape, res[2].shape)
# 3 (2, 4, 35, 3) (2, 4, 35, 3) (2, 4, 35, 2)


