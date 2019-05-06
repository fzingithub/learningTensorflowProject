import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.range(10)   # <tf.Tensor: id=389, shape=(10,), dtype=int32, numpy=array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])>
tf.maximum(a, 2)   # <tf.Tensor: id=581, shape=(10,), dtype=int32, numpy=array([2, 2, 2, 3, 4, 5, 6, 7, 8, 9])>
tf.minimum(a, 2)   # <tf.Tensor: id=635, shape=(10,), dtype=int32, numpy=array([0, 1, 2, 2, 2, 2, 2, 2, 2, 2])>

tf.clip_by_value(a, 2, 8)   # <tf.Tensor: id=530, shape=(10,), dtype=int32, numpy=array([2, 2, 2, 3, 4, 5, 6, 7, 8, 8])>


a = a - 5
# <tf.Tensor: id=750, shape=(10,), dtype=int32, numpy=array([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4])>
tf.nn.relu(a)
# <tf.Tensor: id=869, shape=(10,), dtype=int32, numpy=array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4])>
tf.maximum(a,0)
# <tf.Tensor: id=935, shape=(10,), dtype=int32, numpy=array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4])>


# clip by norm
a = tf.random.normal([2,2], mean=10)
# <tf.Tensor: id=1008, shape=(2, 2), dtype=float32, numpy=
# array([[ 9.298348, 11.598914],
#        [10.152704, 10.486983]], dtype=float32)>

tf.norm(a)   # <tf.Tensor: id=1149, shape=(), dtype=float32, numpy=20.833826>

aa = tf.clip_by_norm(a,15)
tf.norm(aa)   # <tf.Tensor: id=1244, shape=(), dtype=float32, numpy=15.0>