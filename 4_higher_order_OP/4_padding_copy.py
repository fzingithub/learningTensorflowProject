import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# pad
a = tf.reshape(tf.range(9), [3,3])

a
# <tf.Tensor: id=8, shape=(3, 3), dtype=int32, numpy=
# array([[0, 1, 2],
#        [3, 4, 5],
#        [6, 7, 8]])>

tf.pad(a, [[0,0], [0,0]])
# <tf.Tensor: id=8, shape=(3, 3), dtype=int32, numpy=
# array([[0, 1, 2],
#        [3, 4, 5],
#        [6, 7, 8]])>


tf.pad(a, [[1,0], [0,0]])
# <tf.Tensor: id=16, shape=(4, 3), dtype=int32, numpy=
# array([[0, 0, 0],
#        [0, 1, 2],
#        [3, 4, 5],
#        [6, 7, 8]])>


tf.pad(a, [[1,1],[0,0]])
# <tf.Tensor: id=28, shape=(5, 3), dtype=int32, numpy=
# array([[0, 0, 0],
#        [0, 1, 2],
#        [3, 4, 5],
#        [6, 7, 8],
#        [0, 0, 0]])>

tf.pad(a, [[1,1],[1,0]])
# <tf.Tensor: id=44, shape=(5, 4), dtype=int32, numpy=
# array([[0, 0, 0, 0],
#        [0, 0, 1, 2],
#        [0, 3, 4, 5],
#        [0, 6, 7, 8],
#        [0, 0, 0, 0]])>

tf.pad(a, [[1,1],[1,1]])
# <tf.Tensor: id=63, shape=(5, 5), dtype=int32, numpy=
# array([[0, 0, 0, 0, 0],
#        [0, 0, 1, 2, 0],
#        [0, 3, 4, 5, 0],
#        [0, 6, 7, 8, 0],
#        [0, 0, 0, 0, 0]])>


a = tf.random.normal([4,28,28,3])

b = tf.pad(a, [[0,0],[1,1],[1,1],[0,0]])


# tile
a = tf.reshape(tf.range(9), [3,3])

tf.tile(a, [1,2])
# <tf.Tensor: id=37, shape=(3, 6), dtype=int32, numpy=
# array([[0, 1, 2, 0, 1, 2],
#        [3, 4, 5, 3, 4, 5],
#        [6, 7, 8, 6, 7, 8]])>

tf.tile(a, [2,1])
# <tf.Tensor: id=46, shape=(6, 3), dtype=int32, numpy=
# array([[0, 1, 2],
#        [3, 4, 5],
#        [6, 7, 8],
#        [0, 1, 2],
#        [3, 4, 5],
#        [6, 7, 8]])>
tf.tile(a, [2,2])
# <tf.Tensor: id=59, shape=(6, 6), dtype=int32, numpy=
# array([[0, 1, 2, 0, 1, 2],
#        [3, 4, 5, 3, 4, 5],
#        [6, 7, 8, 6, 7, 8],
#        [0, 1, 2, 0, 1, 2],
#        [3, 4, 5, 3, 4, 5],
#        [6, 7, 8, 6, 7, 8]])>


# tile vs broadcast
a = tf.reshape(tf.range(9), [3,3])
aa = tf.expand_dims(a, axis=0)

tf.tile(aa, [2,1,1])
# <tf.Tensor: id=209, shape=(2, 3, 3), dtype=int32, numpy=
# array([[[0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]],
#        [[0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]]])>

tf.broadcast_to(aa, [2,3,3])
# <tf.Tensor: id=272, shape=(2, 3, 3), dtype=int32, numpy=
# array([[[0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]],
#        [[0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]]])>

tf.broadcast_to(a, [2,3,3])
# <tf.Tensor: id=308, shape=(2, 3, 3), dtype=int32, numpy=
# array([[[0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]],
#        [[0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]]])>
