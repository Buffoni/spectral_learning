import tensorflow as tf
import numpy as np
from SpectralLayer import *
from tensorflow.python.compiler.mlcompute import mlcompute

mlcompute.set_mlc_device(device_name='any')

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0 , x_test / 255.0

flat_train = np.reshape(x_train,[x_train.shape[0],28*28])
flat_test = np.reshape(x_test,[x_test.shape[0],28*28])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(28*28,),dtype='float32'))
model.add(tf.keras.layers.Dense(200, activation='tanh'))
model.add(tf.keras.layers.Dense(100, activation='tanh'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

opt = tf.keras.optimizers.Adamax(learning_rate=0.01)#about 0.05  linear and 0.03 eigvalue/0.01eigvector for the non linear
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

epochs=20
history = model.fit(flat_train, y_train,batch_size=128, verbose=1, epochs=epochs)

tested = model.evaluate(flat_test, y_test, batch_size = 100,verbose=2)
