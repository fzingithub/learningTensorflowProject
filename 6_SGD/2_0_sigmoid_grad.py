import tensorflow as tf 
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.linspace(-10., 10., 10)

with tf.GradientTape() as tape:
	tape.watch(a)
	y = tf.sigmoid(a)


grads = tape.gradient(y, [a])
print('x:', a.numpy())
print('y:', y.numpy())
print('grad:', grads[0].numpy())
