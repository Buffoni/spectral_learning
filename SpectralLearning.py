import numpy as np
import tensorflow as tf
from SpectralLayer import Spectral
from tensorflow.keras.layers import Dense
# Parallel execution's stuff
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_synchronous_execution(False)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

flat_train = np.reshape(x_train, [x_train.shape[0], 28*28])
flat_test = np.reshape(x_test, [x_test.shape[0], 28*28])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(28*28), dtype='float32'))
model.add(Spectral(2000, is_base_trainable=True, is_diag_trainable=True, diag_regularizer='l1', use_bias=False, activation='tanh'))
model.add(Spectral(10, is_base_trainable=True, is_diag_trainable=True, use_bias=False, activation='softmax'))
opt = tf.keras.optimizers.Adam(learning_rate=0.003)

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Input(shape=(28*28), dtype='float32'))
# model.add(Dense(2000, kernel_regularizer='l1', use_bias=False, activation='tanh'))
# model.add(Dense(10, use_bias=False, activation='softmax'))
# opt = tf.keras.optimizers.Adam(learning_rate=0.003)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

epochs = 20
history = model.fit(flat_train, y_train, batch_size=800, epochs=epochs)
print('Evaluating on test set...')
testacc = model.evaluate(flat_test, y_test, batch_size=1000, verbose=1)
#%%
eig_number = model.layers[0].diag.numpy().shape[0] + 1

# print('Trim Neurons based on eigenvalue ranking...')
# cut = [0.0, 0.001, 0.01, 0.1, 1]
# for c in cut:
#     zero_out = 0
#     for z in range(0, len(model.layers) - 1):  # put to zero eigenvalues that are below threshold
#         diag_out = model.layers[z].diag.numpy()
#         diag_out[abs(diag_out) < c] = 0
#         model.layers[z].diag = tf.Variable(diag_out)
#         zero_out = zero_out + np.count_nonzero(diag_out == 0)
#     model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     testacc = model.evaluate(flat_test, y_test, batch_size=1000, verbose=0)
#     trainacc = model.evaluate(flat_train, y_train, batch_size=1000, verbose=0)
#     print('Test Acc:', testacc[1], 'Train Acc:', trainacc[1], 'Active Neurons:', 2000-zero_out)
