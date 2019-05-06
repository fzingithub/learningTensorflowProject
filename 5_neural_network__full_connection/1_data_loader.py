# -*- coding: utf-8 -*-
'''
Created on 2019/4/24
Author: zhe
Email: 1194585271@qq.com
'''
import tensorflow as tf
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# MNIST
(x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
print(
    'train data shape:',
    x.shape,   # (60000, 28, 28)
    y.shape,   # (60000,)
    '\ntest data shape:',
    x_test.shape,   # (10000, 28, 28)
    y_test.shape)   # (10000,)

print(
    'min:', x.min(),   # 0
    '\nmax:', x.max(),   # 255
    '\nmean:', x.mean())   # 33.318421449829934

print(y[:4])   # [5 0 4 1]

y_onehot = tf.one_hot(y, depth=10)
print(y_onehot[:2])
# tf.Tensor(
# [[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]], shape=(2, 10), dtype=float32)


# CIFAR10/100
(x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(
    'train data shape:',
    x.shape,   # (50000, 32, 32, 3)
    y.shape,   # (50000, 1)
    '\ntest data shape:',
    x_test.shape,   # (10000, 32, 32, 3)
    y_test.shape)   # (10000, 1)

print(
    'min:', x.min(),   # 0
    '\nmax:', x.max(),   # 255
    '\nmean:', x.mean())   # 120.70756512369792

print(y[:4])
# [[6]
#  [9]
#  [9]
#  [4]]

print(
    type(x),   # <class 'numpy.ndarray'>
    type(y),   # <class 'numpy.ndarray'>
)


# from_tensor_slices()
(x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
# x.shape: (50000, 32, 32, 3)
db = tf.data.Dataset.from_tensor_slices(x_test)
print(next(iter(db)).shape)   # (32, 32, 3)

db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
print(next(iter(db))[0].shape)   # (32, 32, 3)
print (type(next(iter(db))[0]))   # <class 'tensorflow.python.framework.ops.EagerTensor'>

# .shuff
(x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db = db.shuffle(buffer_size=1000)


# .map
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


(x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db2 = db.map(preprocess)

res = next(iter(db2))

print(
    'x_shape:',
    res[0].shape,   # (32, 32, 3)
    '\ny_shape:',
    res[1].shape,   # (1, 10)  注意这和我们预想的不符， 由于载入的 y 有两个维度。
)
print(tf.squeeze(res[1]).shape)   # (10,)


# .batch
(x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db2 = db.map(preprocess)

db3 = db2.batch(32)
res = next(iter(db3))

print(
    'x_shape:',
    res[0].shape,   # x_shape: (32, 32, 32, 3)
    '\ny_shape:',
    res[1].shape,   # y_shape: (32, 1, 10)  ==注意==
)



# StopIteration
(x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db2 = db.map(preprocess)
db3 = db2.batch(32)

db_iter = iter(db3)

while True:
    try:
        next(db_iter)
    except StopIteration:
        print ('StopIteration and break')
        break


# .repeat
(x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db2 = db.map(preprocess)
db3 = db2.batch(1000)
db4 = db3.repeat(2)

for i, (_, _) in enumerate(db4):
    print (i, )
