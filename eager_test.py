from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.enable_eager_execution()


if __name__ == '__main__':
    y = tf.constant([[1, 2], [2, 3], [3, 4]])
    print(y.shape)
    y = tf.tile(y, multiples=[1, 10])
    y = tf.reshape(y, shape=(-1, 2))
    print(y.shape)
    print(y)