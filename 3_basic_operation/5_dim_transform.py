import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# reshape
a = tf.random.normal([4, 28, 28, 3])
a.shape
a.ndim

tf.reshape(a, [4, 28 * 28, 3]).shape
tf.reshape(a, [4, -1, 3]).shape

tf.reshape(a, [4, 784 * 3]).shape
tf.reshape(a, [4, -1]).shape

tf.reshape(tf.reshape(a, [4, -1]), [4, 28, 28, 3]).shape
tf.reshape(tf.reshape(a, [4, -1]), [4, -1, 3]).shape


# transpose

a = tf.random.normal([4, 3, 2, 1])
a.shape
tf.transpose(a).shape
a = tf.random.normal([4, 28, 28, 3])
a.shape
# tensorflow tensor to pytorch tensor
tf.transpose(a, perm=[0, 3, 1, 2]).shape


# expand dim

a = tf.ones([4, 35, 10])
a.shape
tf.expand_dims(a, axis=0).shape  # front
tf.expand_dims(a, axis=-1).shape  # behind
tf.expand_dims(a, axis=-2).shape  # behind

# squeeze dim
a = tf.ones([1, 4, 1, 35, 1, 10])
tf.squeeze(a).shape
