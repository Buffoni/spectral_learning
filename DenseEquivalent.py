import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

flat_train = np.reshape(x_train, [x_train.shape[0], 28*28])
flat_test = np.reshape(x_test, [x_test.shape[0], 28*28])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(28*28), dtype='float32'))
model.add(Dense(2000, use_bias=False, kernel_regularizer='l1', activation='tanh'))
model.add(Dense(10, use_bias=False, activation='softmax'))
opt = tf.keras.optimizers.Adam(learning_rate=0.003)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

epochs = 10
history = model.fit(flat_train, y_train, batch_size=1000, epochs=epochs)
print('Evaluating on test set...')
testacc = model.evaluate(flat_test, y_test, batch_size=1000)

print('Trim Neurons based connectivity ranking...')

# TODO: implement connectivity ranking