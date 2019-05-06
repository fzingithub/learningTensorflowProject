import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# [4, 32, 32, 3]
# + [3]
# + [32, 32, 1]
# + [4, 1, 1, 1]

a = tf.random.normal([4, 32, 32, 3])
b = tf.ones([3])
c = tf.fill([32, 32, 1], 2.)
d = tf.random.uniform([4, 1, 1, 1])

print ((a+b).shape)
print ((a+c).shape)
print ((a+d).shape)


# tf.broadcast_to

x = tf.ones([4, 32, 32, 3])
y = tf.ones([1, 32, 1 ])

print ((x+y).shape)
print (tf.broadcast_to(y, x.shape).shape)
print (tf.ones_like(x).shape)


# Broadcast VS Tile
a = tf.ones([3,4])
a1 = tf.broadcast_to(a, [2,3,4])
print (a1.shape)

a2 = tf.expand_dims(a, axis=0)
a2 = tf.tile(a2, [2,1,1])  # 每一维的倍数
print (a2.shape)