# -*- coding: utf-8 -*-
'''
Created on 2019/6/3
Author: zhe
Email: 1194585271@qq.com
'''

import  tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y


batchsz = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('datasets:', x.shape, y.shape, x.min(), x.max())



db = tf.data.Dataset.from_tensor_slices((x,y))
db = db.map(preprocess).shuffle(60000).batch(batchsz)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsz)

sample = next(iter(db))
print(sample[0].shape, sample[1].shape)


network = Sequential([layers.Dense(256, activation='relu'),
                     layers.Dense(128, activation='relu'),
                     layers.Dense(64, activation='relu'),
                     layers.Dense(32, activation='relu'),
                     layers.Dense(10)])
network.build(input_shape=(None, 28*28))
network.summary()


custom_loss = tf.losses.CategoricalCrossentropy

network.compile(optimizer=optimizers.Adam(lr=0.01),
		loss=tf.losses.CategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
	)

network.fit(db, epochs=3, validation_data=ds_val, validation_freq=2)

network.evaluate(ds_val)

network.save('./save_model/model.h5')
print('saved total model.')
del network



print('load model from file')
network = tf.keras.models.load_model('./save_model/model.h5')
network.build(input_shape=(None, 28 * 28))

x_val = tf.cast(x_val, dtype=tf.float32) / 255.
x_val = tf.reshape(x_val, [-1, 28*28])
y_val = tf.cast(y_val, dtype=tf.int32)
result = network(x_val)

result = tf.argmax(result, axis=-1)
result = tf.cast(result, dtype=tf.int32)
accuracy = tf.reduce_sum(tf.cast(tf.equal(result, y_val), dtype=tf.int32))/y_val.shape[0]

print('accuracy:', float(accuracy))

