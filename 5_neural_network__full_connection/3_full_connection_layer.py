# -*- coding: utf-8 -*-
'''
Created on 2019/4/30
Author: zhe
Email: 1194585271@qq.com
'''


import tensorflow as tf
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# full connection
x = tf.random.normal([4, 784])
net = tf.keras.layers.Dense(512)
out = net(x)

print(out.shape)   # (4, 512)
print(net.kernel.shape, net.bias.shape)   # (784, 512) (512,)


net = tf.keras.layers.Dense(10)
#print(net.bias)   # AttributeError: 'Dense' object has no attribute 'bias'

print(net.get_weights())   # []
print(net.weights)   # []

net.build(input_shape=[None, 4])
print(net.kernel.shape, net.bias.shape)   # (4, 10) (10,)

net.build(input_shape=[None, 20])
print(net.kernel.shape, net.bias.shape)   # (20, 10) (10,)

net.build(input_shape=[2, 4])
print(net.kernel.shape, net.bias.shape)   # (4, 10) (10,)
print (net.kernel)




net.build(input_shape=[None, 20])
print(net.kernel.shape, net.bias.shape)   # (20, 10) (10,)

# out = net(tf.random.normal([4, 12]))   # InvalidArgumentError: Matrix size-incompatible: In[0]: [4,12], In[1]: [20,10] [Op:MatMul]

out = net(tf.random.normal([4, 20]))   # (20, 10) (10,)
print(out.shape)   # (4, 10)

x = tf.random.normal([2, 3])

model = keras.Sequential([
    keras.layers.Dense(2, activation='relu'),
    keras.layers.Dense(2, activation='relu'),
    keras.layers.Dense(2)
])

model.build(input_shape=[2, 3])
model.summary()

for p in model.trainable_weights:
    print(p.name, p.shape)


out = model(x)
print(out.shape)   # (2, 2)
