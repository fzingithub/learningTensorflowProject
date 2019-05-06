import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.fill([2,2], 3.)
b = tf.fill([2,2], 2.)

# 加减乘除
a+b, a-b, a*b, a/b
# (<tf.Tensor: id=26, shape=(2, 2), dtype=int32, numpy=
#  array([[5, 5],
#         [5, 5]])>, <tf.Tensor: id=27, shape=(2, 2), dtype=int32, numpy=
#  array([[1, 1],
#         [1, 1]])>, <tf.Tensor: id=28, shape=(2, 2), dtype=int32, numpy=
#  array([[6, 6],
#         [6, 6]])>, <tf.Tensor: id=31, shape=(2, 2), dtype=float64, numpy=
#  array([[1.5, 1.5],
#         [1.5, 1.5]])>)
a//b, a%b
# (<tf.Tensor: id=12, shape=(2, 2), dtype=int32, numpy=
#  array([[1, 1],
#         [1, 1]])>, <tf.Tensor: id=13, shape=(2, 2), dtype=int32, numpy=
#  array([[1, 1],
#         [1, 1]])>)

# 指数对数
tf.math.log(a)
# <tf.Tensor: id=118, shape=(2, 2), dtype=float32, numpy=
# array([[1.0986123, 1.0986123],
#        [1.0986123, 1.0986123]], dtype=float32)>

tf.exp(a)
# <tf.Tensor: id=164, shape=(2, 2), dtype=float32, numpy=
# array([[20.085537, 20.085537],
#        [20.085537, 20.085537]], dtype=float32)>

tf.math.log(8.)/tf.math.log(2.)



# pow , sqrt
tf.pow(b, 3)
# <tf.Tensor: id=514, shape=(2, 2), dtype=float32, numpy=
# array([[8., 8.],
#        [8., 8.]], dtype=float32)>
b**3
# <tf.Tensor: id=573, shape=(2, 2), dtype=float32, numpy=
# array([[8., 8.],
#        [8., 8.]], dtype=float32)>
tf.sqrt(b)
# <tf.Tensor: id=634, shape=(2, 2), dtype=float32, numpy=
# array([[1.4142135, 1.4142135],
#        [1.4142135, 1.4142135]], dtype=float32)>

a,b
# (<tf.Tensor: id=64, shape=(2, 2), dtype=float32, numpy=
#  array([[3., 3.],
#         [3., 3.]], dtype=float32)>,
#  <tf.Tensor: id=67, shape=(2, 2), dtype=float32, numpy=
#  array([[2., 2.],
#         [2., 2.]], dtype=float32)>)
tf.matmul(a,b)
# <tf.Tensor: id=925, shape=(2, 2), dtype=float32, numpy=
# array([[12., 12.],
#        [12., 12.]], dtype=float32)>
a @ b
# <tf.Tensor: id=1008, shape=(2, 2), dtype=float32, numpy=
# array([[12., 12.],
#        [12., 12.]], dtype=float32)>


a = tf.ones([4,1,2])
b = tf.fill([4,2,2], 2.)

(a@b).shape


a = tf.ones([4,5,6])
b = tf.fill([6,4], 2.)
bb = tf.broadcast_to(b, [4,6,4])
(a@bb).shape






# 𝑌 = 𝑋@𝑊 + 𝑏
x = tf.ones([4, 2])
W = tf.ones([2, 1])
b = tf.constant(0.1)
Y = x@W + b

out = x@W + b

out = tf.nn.relu(out)
# <tf.Tensor: id=1953, shape=(4, 1), dtype=float32, numpy=
# array([[2.1],
#        [2.1],
#        [2.1],
#        [2.1]], dtype=float32)>
