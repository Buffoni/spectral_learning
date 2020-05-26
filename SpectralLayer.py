#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of a simple Spectral NN layer that can learn eigenvalues
and eigenvectors of the associated adjacency matrix.

@authors: Lorenzo Buffoni, Lorenzo Giambagli
"""

import numpy as np
import tensorflow as tf


class SpectralLayer(tf.keras.layers.Layer):
    def __init__(self, s_init, s_final, layer_dim, next_layer_dim, dim,
                 activation=None, is_base_trainable=True, is_diag_trainable=True):
        super(SpectralLayer, self).__init__()
        """Instantiate the eigenvalues and eigenvectors matrices using the proper dimensions.
        Also builds the appropriate masks to stop backpropagation in undesired regions.
        Parameters
        ----------
        s_init: (int) initial idex of the eigenvector block
        s_final: (int) final index of the eigenvector block
        layer_dim: (int) dimension of current block
        next_layer_dim: (int) dimension of next block
        dim: (int) total dimension of the network space
        activation: (string) nonlinear activation function either: "sigmoid", "tanh", "relu" or "softmax"
        is_base_trainable: (bool) train eigenvectors
        is_diag_trainable: (bool) train eigenvalues
        """
        self.final_shape = next_layer_dim
        # construct the indented base (random blocks + diagonal)
        bases = np.zeros([dim, dim])
        bases[s_init:s_final, s_init - layer_dim:s_init] = np.random.uniform(-0.2, 0.2, (next_layer_dim, layer_dim))
        b_mask = np.zeros([dim, dim])
        b_mask[s_init:s_final, s_init - layer_dim:s_init] = np.ones((next_layer_dim, layer_dim))
        self.base_mask = tf.constant(b_mask, dtype=tf.float32)
        if is_base_trainable:
            self.base = tf.Variable(bases, trainable=True, dtype=tf.float32)
        else:
            self.base = tf.constant(bases, dtype=tf.float32)
        self.eye = tf.constant(np.identity(dim), dtype=tf.float32)

        # construct the diagonal eigenvalues matrix (random initialized)
        diags = np.identity(dim)
        masks = np.zeros([dim, dim])
        masks[s_init:s_final, s_init:s_final] = np.diag(
            np.random.uniform(-1.8, 1.8, (next_layer_dim,)))
        diags = diags - masks
        self.untrainable_diag = tf.constant(diags, dtype=tf.float32)
        self.trainable_diag_mask = tf.constant(masks, dtype=tf.float32)
        if is_diag_trainable:
            self.trainable_diag = tf.Variable(masks, trainable=True, dtype=tf.float32)
        else:
            self.trainable_diag = tf.constant(masks, dtype=tf.float32)

        # mask for unactivated outputs
        mask = np.ones((dim, 1))
        mask[s_init:s_final] = np.zeros((s_final - s_init, 1))
        self.mask1 = tf.constant(mask, dtype=tf.float32)

        # mask for activated outputs
        mask = np.zeros((dim, 1))
        mask[s_init:s_final] = np.ones((s_final - s_init, 1))
        self.mask2 = tf.constant(mask, dtype=tf.float32)

        self.offset = tf.constant(np.ones([dim, 1]), dtype=tf.float32)

        if activation == 'relu':
            self.nonlinear = tf.nn.relu
            self.last = False
        elif activation == 'sigmoid':
            self.nonlinear = tf.math.sigmoid
            self.last = False
        elif activation == 'tanh':
            self.nonlinear = tf.math.tanh
            self.last = False
        elif activation == 'softmax':
            self.nonlinear = tf.nn.softmax
            self.last = True
        else:
            self.nonlinear = None
            self.last = False

    def call(self, data, training=False):
        """Performs the spectral layer operation.
        Parameters
        ----------
        data: (float) input data or output from a previous layer
                
        Returns
        -------
        x: (tf.float32) output of a single spectral layer
        """
        layer = tf.math.add(tf.math.multiply(self.base, self.base_mask), self.eye)
        diag = tf.math.add(tf.math.multiply(self.trainable_diag, self.trainable_diag_mask), self.untrainable_diag)
        x = tf.linalg.matmul(tf.math.subtract(2 * self.eye, layer), tf.transpose(data)) #2*I-layer is the analytical inverse of layer
        x = tf.linalg.matmul(diag, x)
        x = tf.linalg.matmul(layer, x)

        if self.nonlinear is not None:
            temp = tf.transpose(self.nonlinear(tf.transpose(x)))
        else:
            temp = x

        x = tf.math.multiply(x, self.mask1)
        temp = tf.math.multiply(temp, self.mask2)
        x = tf.math.add(x, temp)
        if self.last:
            x = x[x.shape[0] - self.final_shape:]
        x = tf.transpose(x)
        return x


if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf

    mylayer = SpectralLayer(784,794,784,10,794,activation='softmax')
    test_tensor = tf.constant(np.ones((2, 794)), dtype=tf.float32)
    out = mylayer(test_tensor)
    print("If test succeds output shape should be: (2, 10)")
    print("Output shape: ",out.shape)
