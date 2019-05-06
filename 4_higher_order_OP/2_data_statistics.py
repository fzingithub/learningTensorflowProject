import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


a = tf.ones([2,2])
tf.norm(a)   # <tf.Tensor: id=100, shape=(), dtype=float32, numpy=2.0>
tf.sqrt(tf.reduce_sum(tf.square(a)))   # <tf.Tensor: id=81, shape=(), dtype=float32, numpy=2.0>

a = tf.ones([4,28,28,3])
tf.norm(a)   # <tf.Tensor: id=48, shape=(), dtype=float32, numpy=96.99484>
tf.sqrt(tf.reduce_sum(tf.square(a)))   # <tf.Tensor: id=59, shape=(), dtype=float32, numpy=96.99484>

b = tf.ones([4,28,28,3])
tf.norm(b).shape

tf.norm(b, ord=2, axis=0).shape   # ord 2范数   # Out[4]: TensorShape([28, 28, 3])
tf.norm(b, ord=2, axis=1).shape   # TensorShape([4, 28, 3])
tf.norm(b, ord=1).shape   # TensorShape([])

# Reduce_min\max\mean
a = tf.random.normal([4,10])

tf.reduce_min(a), tf.reduce_max(a), tf.reduce_mean(a)
# (<tf.Tensor: id=77, shape=(), dtype=float32, numpy=-1.8365537>,
#  <tf.Tensor: id=79, shape=(), dtype=float32, numpy=2.47236>,
#  <tf.Tensor: id=81, shape=(), dtype=float32, numpy=0.08888846>)

tf.reduce_min(a, axis=1).shape   # TensorShape([4])
tf.reduce_max(a, axis=0).shape   # TensorShape([10])



# Argmax/Argmin
a = tf.random.normal([4,10])
a.shape   # TensorShape([4, 10])

tf.argmax(a)   # <tf.Tensor: id=455, shape=(10,), dtype=int64, numpy=array([1, 2, 0, 2, 0, 1, 0, 1, 1, 2], dtype=int64)>
tf.argmax(a, axis=1)   # <tf.Tensor: id=378, shape=(4,), dtype=int64, numpy=array([2, 8, 3, 1], dtype=int64)>
tf.argmin(a, axis=1)   # <tf.Tensor: id=415, shape=(4,), dtype=int64, numpy=array([7, 2, 8, 2], dtype=int64)>


# Equal
a = tf.constant([2, 1, 3, 2, 5])
b = tf.range(5)
res = tf.equal(a, b)

tf.reduce_sum(tf.cast(res, dtype=tf.int32))   # <tf.Tensor: id=557, shape=(), dtype=int32, numpy=1>

# Accuracy
pre = tf.constant([[0.1, 0.2, 0.7],[0.9, 0.05, 0.05]])
pre
# <tf.Tensor: id=1468, shape=(2, 3), dtype=float32, numpy=
# # array([[0.1 , 0.2 , 0.7 ],
# #        [0.9 , 0.05, 0.05]], dtype=float32)>
pre = tf.cast(tf.argmax(pre, axis=1), dtype=tf.int32)
# <tf.Tensor: id=1205, shape=(2,), dtype=int32, numpy=array([2, 0])>

y = tf.constant([2,1])   # <tf.Tensor: id=1206, shape=(2,), dtype=int32, numpy=array([2, 1])>

accuracy = tf.reduce_sum(tf.cast(tf.equal(pre, y), dtype=tf.int32))/pre[0]
# <tf.Tensor: id=1218, shape=(), dtype=float64, numpy=0.5>



# Unique Gather

a = tf.range(5)
print (a)

unique_tensor, idx =  tf.unique(a)
print (unique_tensor, idx)

a = tf.constant([4,2,2,1,4,0])
a   # <tf.Tensor: id=170, shape=(6,), dtype=int32, numpy=array([4, 2, 2, 1, 4, 0])>

unique_tensor, idx =  tf.unique(a)
unique_tensor, idx
# (<tf.Tensor: id=172, shape=(4,), dtype=int32, numpy=array([4, 2, 1, 0])>,
#  <tf.Tensor: id=173, shape=(6,), dtype=int32, numpy=array([0, 1, 1, 2, 0, 3])>)

origin_a = tf.gather(unique_tensor, idx)
# <tf.Tensor: id=216, shape=(6,), dtype=int32, numpy=array([4, 2, 2, 1, 4, 0])>