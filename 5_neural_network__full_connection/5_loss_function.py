# -*- coding: utf-8 -*-
'''
Created on 2019/5/6
Author: zhe
Email: 1194585271@qq.com
'''
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



# mse
y = tf.constant([1, 2, 3, 0, 2])
y = tf.one_hot(y, depth=4)
y = tf.cast(y, dtype=tf.float32)

out = tf.random.normal([5, 4])


loss1 = tf.reduce_mean(tf.square(y-out))

loss2 = tf.square(tf.norm(y-out))/(5*4)

loss3 = tf.reduce_mean(tf.losses.MSE(y, out)) # VS MeanSquaredError is a class

print(loss1)
print(loss2)
print(loss3)



# cross entropy
a = tf.fill([4], 0.25)
print(a*tf.math.log(a)/tf.math.log(2.))
# tf.Tensor([-0.5 -0.5 -0.5 -0.5], shape=(4,), dtype=float32)
print(-tf.reduce_sum(a*tf.math.log(a)/tf.math.log(2.)))
# tf.Tensor(2.0, shape=(), dtype=float32)

a = tf.constant([0.1, 0.1, 0.1, 0.7])
print(-tf.reduce_sum(a*tf.math.log(a)/tf.math.log(2.)))
# tf.Tensor(1.3567796, shape=(), dtype=float32)

a = tf.constant([0.01, 0.01, 0.01, 0.97])
print(-tf.reduce_sum(a*tf.math.log(a)/tf.math.log(2.)))
# tf.Tensor(0.24194068, shape=(), dtype=float32)

# categorical cross entropy
print(tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.25, 0.25, 0.25, 0.25]))
# tf.Tensor(1.3862944, shape=(), dtype=float32)
print(tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.1, 0.1, 0.8, 0.1]))
# tf.Tensor(2.3978953, shape=(), dtype=float32)
print(tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.1, 0.7, 0.1, 0.1]))
# tf.Tensor(0.35667497, shape=(), dtype=float32)
print(tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.01, 0.97, 0.01, 0.01]))
# tf.Tensor(0.030459179, shape=(), dtype=float32)

print(tf.losses.CategoricalCrossentropy()([0, 1, 0, 0], [0.01, 0.97, 0.01, 0.01]))
# tf.Tensor(0.030459179, shape=(), dtype=float32)


# binary cross entropy
print(tf.losses.CategoricalCrossentropy()([0, 1], [0.03, 0.97]))
# tf.Tensor(0.030459179, shape=(), dtype=float32)
print(tf.losses.categorical_crossentropy([0, 1], [0.03, 0.97]))
# tf.Tensor(0.030459179, shape=(), dtype=float32)

print(tf.losses.BinaryCrossentropy()([1], [0.97]))
# tf.Tensor(0.030459056, shape=(), dtype=float32)
print(tf.losses.binary_crossentropy([1], [0.97]))
# tf.Tensor(0.030459056, shape=(), dtype=float32)


# numerical stability
x = tf.random.normal([1, 784])
w = tf.random.normal([784, 2])
b = tf.zeros([2])

logits = x@w+b

prob = tf.math.softmax(logits)

print(tf.losses.categorical_crossentropy([0, 1], logits, from_logits=True))  # 数值稳定
# tf.Tensor([0.], shape=(1,), dtype=float32)
print(tf.losses.categorical_crossentropy([0, 1], prob))   # 不推荐
# tf.Tensor([1.192093e-07], shape=(1,), dtype=float32)