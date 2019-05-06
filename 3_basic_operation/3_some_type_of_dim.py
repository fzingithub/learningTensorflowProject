import tensorflow as tf
from tensorflow.keras import layers
import keras
from keras.layers import Embedding
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# scalar  --> loss
out = tf.random.uniform([4,10])

y = tf.range(4)
y = tf.one_hot(y, depth=10)

loss = tf.keras.losses.mse(y, out)
loss = tf.reduce_mean(loss)
print (loss)


# vector
net = layers.Dense(10)
net.build([4, 8])
print (net.kernel.shape)  # matrix
print (net.bias)   # vector

# matrix
x = tf.random.normal([4, 784])
net = layers.Dense(10)   # 784 --> 10
net.build([4, 784])

print (net(x).shape)


# tensor dim = 3
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)
x_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=80)
print(x_train.shape)

# emb = embedding(x_train)
# print (emb.shape)
#
# out = rnn(emb[:4])
# print (out.shape)



# tensor dim=4
x = tf.random.normal([4,28,28,1])
net = layers.Conv2D(16, kernel_size=3)
print (net(x).shape)