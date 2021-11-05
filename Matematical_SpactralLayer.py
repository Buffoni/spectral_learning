#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of a simple Spectral NN layer that can learn eigenvalues
and eigenvectors of the associated adjacency matrix.

reference: https://arxiv.org/abs/2005.14436

@authors: Lorenzo Buffoni, Lorenzo Giambagli
"""

import numpy as np
import tensorflow as tf


class SpectralLayer(tf.keras.layers.Layer):
    def __init__(self, next_layer_dim, activation=None,
                 is_base_trainable=True, is_diag_trainable=True):
        super(SpectralLayer, self).__init__()

        self.final_shape = next_layer_dim
        self.is_base_trainable = is_base_trainable
        self.is_diag_trainable = is_diag_trainable

        if activation == 'relu':
            self.nonlinear = tf.nn.relu
        elif activation == 'sigmoid':
            self.nonlinear = tf.math.sigmoid
        elif activation == 'tanh':
            self.nonlinear = tf.math.tanh
        elif activation == 'softmax':
            self.nonlinear = tf.nn.softmax
        else:
            self.nonlinear = None

    def build(self, input_shape):
        input_shape = input_shape[1]
        self.dim = input_shape + self.final_shape
        self.eye = tf.constant(np.identity(self.dim), dtype=tf.float32)

        # construct the indented base (random block + zero blocks)
        self.zero_block_1 = tf.constant(np.zeros([input_shape, self.dim]), dtype=tf.float32)
        self.zero_block_2 = tf.constant(np.zeros([self.final_shape, self.final_shape]), dtype=tf.float32)
        block = np.random.uniform(-0.2, 0.2, (self.final_shape, input_shape))
        if self.is_base_trainable:
            self.block = tf.Variable(block, trainable=True, dtype=tf.float32)
        else:
            self.block = tf.constant(block, dtype=tf.float32)

        # construct the diagonal eigenvalues matrix
        self.zero_diag = tf.constant(np.zeros([self.final_shape, input_shape]), dtype=tf.float32)
        block_diag = np.random.uniform(-0.5, 0.5, (1, self.final_shape))
        if self.is_diag_trainable:
            self.trainable_diag = tf.Variable(block_diag, trainable=True, dtype=tf.float32)
        else:
            self.trainable_diag = tf.constant(block_diag, dtype=tf.float32)

    def call(self, data, training=False):
        # concatenate the trainable blocks with the constant ones
        data = tf.keras.layers.ZeroPadding1D(padding=(0, self.final_shape))(tf.reshape(data, [-1, data.shape[1], 1]))
        data = tf.reshape(data, [-1, data.shape[1]])
        layer = tf.math.add(self.eye, tf.concat([self.zero_block_1, tf.concat([self.block, self.zero_block_2], 1)], 0))
        eigval = tf.reshape(tf.linalg.diag(self.trainable_diag), [self.final_shape, self.final_shape])
        diag = tf.math.add(self.eye, tf.concat([self.zero_block_1, tf.concat([self.zero_diag, eigval], 1)], 0))
        x = tf.linalg.matmul(tf.math.subtract(2 * self.eye, layer), tf.transpose(data))  # 2*I-layer is the analytical inverse of layer
        x = tf.linalg.matmul(diag, x)
        x = tf.linalg.matmul(layer, x)
        x = x[x.shape[0] - self.final_shape:]
        if self.nonlinear is not None:
            x = self.nonlinear(tf.transpose(x))
        return x


if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf

    mylayer = SpectralLayer(10, activation='softmax')
    test_tensor = tf.constant(np.ones((2, 794)), dtype=tf.float32)
    out = mylayer(test_tensor)
    print("If test succeeds output shape should be: (2, 10)")
    print("Output shape: ", out.shape)
