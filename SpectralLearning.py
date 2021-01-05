import numpy as np
import tensorflow as tf
from SpectralLayer import Spectral
from tensorflow.python.compiler.mlcompute import mlcompute

mlcompute.set_mlc_device(device_name='any')

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

flat_train = np.reshape(x_train, [x_train.shape[0], 28*28])[:2000]
y_train = y_train[:2000]
flat_test = np.reshape(x_test, [x_test.shape[0], 28*28])[:2000]
y_test =y_test[:2000]


model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(28*28), dtype='float32'))
model.add(Spectral(2000, is_base_trainable=True, is_diag_trainable=True, use_bias=False, activation='tanh'))
model.add(Spectral(10, is_base_trainable=True, is_diag_trainable=True, use_bias=False, activation='softmax'))
opt = tf.keras.optimizers.Adam(learning_rate=0.003)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

epochs = 20
history = model.fit(flat_train, y_train, validation_data=(flat_test, y_test), batch_size=100, epochs=epochs)
print('Evaluating on test set...')
testacc = model.evaluate(flat_test, y_test, batch_size=100)
acc_0 = testacc[1]
eig_number = model.layers[0].diag.numpy().shape[0] + 10


print('Trim Neurons based on eigenvalue ranking...')
cut = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# cut = [0.9, 0.5, 0.4, 0.3, 0.2, 0.1, 0.07, 0.05, 0.02, 0.0]
# cut = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
for c in cut:
    zero_out = 0
    for z in range(0, len(model.layers) - 1):  # Azzero autovalori minori di soglia cut
        diag_out = model.layers[z].diag.numpy()
        diag_out[abs(diag_out) < c] = 0
        model.layers[z].diag = tf.Variable(diag_out)
        zero_out = zero_out + np.count_nonzero(diag_out == 0)

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.save_weights('./test')
    testacc = model.evaluate(flat_test, y_test, batch_size=1000, verbose=0)
    trainacc = model.evaluate(flat_train, y_train, batch_size=1000, verbose=0)
    print('Test Acc:', testacc[1], 'Train Acc:', trainacc[1], 'Active Neurons:', 2000-zero_out)

'''
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(28*28), dtype='float32'))
model.add(Spectral(5000, is_base_trainable=True, is_diag_trainable=False, use_bias=False, activation='tanh'))
model.add(Spectral(10, is_base_trainable=True, is_diag_trainable=False, use_bias=False, activation='softmax'))

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.load_weights('./test')
epochs = 10
history = model.fit(flat_train, y_train, batch_size=1000, epochs=epochs)
print('Evaluating on test set...')
testacc = model.evaluate(flat_test, y_test, batch_size=1000)
'''
