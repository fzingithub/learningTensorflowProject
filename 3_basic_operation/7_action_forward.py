import tensorflow as tf
from tensorflow.keras import datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load minst data
# x: [60k, 28, 28],
# y: [60k]
(x, y), _ = datasets.mnist.load_data()
# x: [0~255] => [0~1.]
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)

print (x.shape, y.shape, x.dtype, y.dtype)
print (tf.reduce_min(x), tf.reduce_max(x))
print (tf.reduce_min(y), tf.reduce_max(y))

batch_size = 128
train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(batch_size)
train_iter = iter(train_db)
sample = next(train_iter)
print ('x_batch_size:',sample[0].shape,  'y_batch_size:', sample[1].shape)




# [b, 784] => [b, 256] => [b, 128] => [b, 10]
# w:[in_dim, out_dim], b:[dim_out]

w1 = tf.Variable(tf.random.truncated_normal([784, 512], stddev=0.05))  # stddev 设置标准差 防止梯度弥散
b1 = tf.Variable(tf.zeros([512]))
w2 = tf.Variable(tf.random.truncated_normal([512, 256], stddev=0.05))
b2 = tf.Variable(tf.zeros([256]))
w3 = tf.Variable(tf.random.truncated_normal([256, 10], stddev=0.05))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3

for epoch in range(10):
    for step, (x, y) in enumerate(train_db):
        # x:[b, 28, 28]
        # y: [b]

        # [b, 28, 28] => [b, 28*28]
        x = tf.reshape(x, [-1,28*28])


        with tf.GradientTape() as tape:  # 默认跟踪 tf.Variable 变量
            # x: [b, 28*28]
            # h1 = x@w1 + b1
            # [b, 784]@[784, 256] + [256] => [b, 256] + [256] => [b, 256] + [b, 256]
            h1 = x@w1+tf.broadcast_to(b1, [x.shape[0], 512])
            h1 = tf.nn.relu(h1)
            # [b, 256] => [b, 128]
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)
            # [b, 128] => [b, 10]
            out = h2@w3 + b3

            # compute loss
            # out: [b, 10]
            # y: [b] => [b, 10]

            y_onehot = tf.one_hot(y, depth=10)

            # mse = mean(sum(y-out)^2)
            # [b, 10]
            loss = tf.square(y_onehot - out)
            loss = tf.reduce_mean(loss)

        # compute gradients
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # print(grads)
        # w1 = w1 - lr * w1_grad
        w1.assign_sub(lr * grads[0])   # 原地更新  继续为 Variable 变量
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step%100 == 0:
            print ('epoch:', epoch, 'step:', step, 'loss:', float(loss))