import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# basic indexing
a = tf.ones([1,5,5,3])

print (a[0][0].shape)

print (a[0][0][0].shape)

print (a[0][0][0][2].shape)


# Numpy-style indexing

b = tf.random.normal([4, 28, 28, 3])

print (b[1].shape)
print (b[1, 2].shape)
print (b[1, 2, 3].shape)
print (b[1, 2, 3, 2].shape)

# start:end

c = tf.range(10)

print (c[-1:])
print (c[-2:])
print (c[:2])
print (c[:-1])


# Indexing by :

d = tf.random.normal([4, 28, 28, 3])

print (d[0].shape)
print (d[0,:,:,:].shape)
print (d[0,1,:,:].shape)
print (d[:,:,:,0].shape)
print (d[:,:,:,2].shape)
print (d[:,0,:,0].shape)


# Indexing by ::
print (d.shape)
print (d[0:2,:,:,:].shape)
print (d[:,14:,14:,:].shape)
print (d[:,0:28:2,0:28:2,:].shape)
print (d[:,::2,::2,:].shape)

# ::-1
e = tf.range(10)
print (e)
print (e[::-1])
print (e[::-2])
print (e[2::-2])


# ...
f = tf.random.normal([2, 4, 28, 28, 3])
print (f[0].shape)
print (f[0,:,:,:,:].shape)
print (f[0,...].shape)

print (f[:,:,:,:,0].shape)
print (f[...,0].shape)

print (f[0,...,0].shape)
print (f[0,2,...,0].shape)


# Selective Indexing
#tf.gather

g = tf.random.normal([4, 35, 8])
print (tf.gather(g, axis=0, indices=[1,3]).shape)
print (g[1::2].shape)
print (tf.reduce_all(tf.equal(tf.gather(g, axis=0, indices=[1,3]), g[1::2])))

print (tf.gather(g, axis=0, indices=[0,3,2,3,1]).shape)
print (tf.gather(g, axis=1, indices=[0,11,25]).shape)
print (tf.gather(g, axis=2, indices=[2,4]).shape)

# tf.gather_nd
print (g.shape)
print (tf.gather_nd(g, [0,1]).shape)
print (tf.gather_nd(g, [0, 1, 2]).shape)
print (tf.gather_nd(g, [[0, 1, 2]]).shape)


print (tf.gather_nd(g, [[0,1], [2,32]]).shape)
print (tf.gather_nd(g, [[0,0,0],[1,2,3],[2,3,2]]).shape)
print (tf.gather_nd(g, [[[0,0,0],[1,2,3],[2,3,2]]]).shape)

# tf.boolean_mask
h = tf.random.normal([4, 28, 28, 3])
print (h.shape)

print (tf.boolean_mask(h, mask=[True, True, False, False]).shape)
print (tf.boolean_mask(h, mask=[True,False,True], axis=3).shape)

i = tf.random.truncated_normal([2,3,4])
# print (i)
print (tf.boolean_mask(i, mask=[[True, False, False],[False,False,True]]))