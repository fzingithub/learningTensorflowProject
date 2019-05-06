import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Sort/argsort
a = tf.random.shuffle(tf.range(5))
# <tf.Tensor: id=25, shape=(5,), dtype=int32, numpy=array([3, 2, 0, 4, 1])>
tf.sort(a, direction='DESCENDING')
# <tf.Tensor: id=13, shape=(5,), dtype=int32, numpy=array([4, 3, 2, 1, 0])>
tf.argsort(a, direction='DESCENDING')
#  <tf.Tensor: id=50, shape=(5,), dtype=int32, numpy=array([3, 0, 1, 4, 2])>

idx = tf.argsort(a, direction='DESCENDING')
a_sorted = tf.gather(a, idx)
# <tf.Tensor: id=76, shape=(5,), dtype=int32, numpy=array([4, 3, 2, 1, 0])>


a = tf.random.uniform([3,3], maxval=10, dtype=tf.int32)
# <tf.Tensor: id=148, shape=(3, 3), dtype=int32, numpy=
# array([[0, 4, 0],
#        [8, 9, 3],
#        [8, 7, 3]])>
tf.sort(a, direction='DESCENDING')
# <tf.Tensor: id=203, shape=(3, 3), dtype=int32, numpy=
# array([[4, 0, 0],
#        [9, 8, 3],
#        [8, 7, 3]])>
tf.argsort(a, direction='DESCENDING')
# <tf.Tensor: id=242, shape=(3, 3), dtype=int32, numpy=
# array([[1, 0, 2],
#        [1, 0, 2],
#        [0, 1, 2]])>

a = tf.random.uniform([3,3], maxval=10, dtype=tf.int32)
# <tf.Tensor: id=1988, shape=(3, 3), dtype=int32, numpy=
# array([[7, 3, 0],
#        [8, 8, 6],
#        [2, 6, 1]])>
tf.sort(a, direction='DESCENDING', axis=0)
# <tf.Tensor: id=2185, shape=(3, 3), dtype=int32, numpy=
# array([[8, 8, 6],
#        [7, 6, 1],
#        [2, 3, 0]])>
tf.argsort(a, direction='DESCENDING', axis=0)
# <tf.Tensor: id=2295, shape=(3, 3), dtype=int32, numpy=
# array([[1, 1, 1],
#        [0, 2, 2],
#        [2, 0, 0]])>


# Top_k
a = tf.random.uniform([3,3], maxval=10, dtype=tf.int32)
# <tf.Tensor: id=2399, shape=(3, 3), dtype=int32, numpy=
# array([[6, 3, 6],
#        [4, 7, 4],
#        [9, 6, 9]])>
res = tf.math.top_k(a, k=1, sorted=True)
res.values
# <tf.Tensor: id=3765, shape=(3, 1), dtype=int32, numpy=
# array([[6],
#        [7],
#        [9]])>
res.indices
# <tf.Tensor: id=3766, shape=(3, 1), dtype=int32, numpy=
# array([[0],
#        [1],
#        [0]])>

# Top_k accuracy
prob = tf.constant([[0.1, 0.2, 0.7], [0.2, 0.7, 0.1]])
target = tf.constant([2, 0])

k_b = tf.math.top_k(prob, k=2).indices

target = tf.broadcast_to(target, [2,2])
# <tf.Tensor: id=5115, shape=(3, 2), dtype=int32, numpy=
# # array([[2, 0],
# #        [2, 0],
# #        [2, 0]])>
correct = tf.equal(target, k_b)
# <tf.Tensor: id=6768, shape=(2, 2), dtype=bool, numpy=
# array([[ True, False],
#        [False,  True]])>
bool2int = tf.cast(correct, dtype=tf.int32)
# <tf.Tensor: id=7115, shape=(2, 2), dtype=int32, numpy=
# array([[1, 0],
#        [0, 1]])>
num_of_pre_correct = tf.reduce_sum(bool2int)
# <tf.Tensor: id=7471, shape=(), dtype=int32, numpy=2>
top_k_accuracy = num_of_pre_correct/target.shape[0]
# <tf.Tensor: id=7837, shape=(), dtype=float64, numpy=1.0>