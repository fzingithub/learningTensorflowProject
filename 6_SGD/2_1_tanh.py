# -*- coding: utf-8 -*-
'''
Created on 2019/5/9
Author: zhe
Email: 1194585271@qq.com
'''

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.linspace(-10., 10., 10)

print(tf.tanh(a))