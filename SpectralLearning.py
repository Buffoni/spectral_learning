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
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
#
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Input(shape=(28*28), dtype='float32'))
# model.add(Dense(2000, kernel_regularizer=tf.keras.regularizers.l1(l1=0.0005), use_bias=False, activation='tanh'))
# model.add(Dense(10, use_bias=False, activation='softmax'))
# opt = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

epochs = 70
history = model.fit(flat_train, y_train, batch_size=500, epochs=epochs)
print('Evaluating on test set...')
testacc = model.evaluate(flat_test, y_test, batch_size=1000, verbose=1)
#%%HYPERPARAMETER OPTIM
from ray import tune
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
# # Parallel execution's stuff
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.experimental.set_synchronous_execution(False)


config = {"l1": tune.grid_search([0.5E-3]),
          "lr": tune.grid_search([1E-2, 1E-3, 5E-3]),
          "epochs": tune.grid_search([40, 70])}

trialResources = {'cpu': 3, 'gpu': 8 / 40}

def train_funct(config):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_synchronous_execution(False)

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    flat_train = np.reshape(x_train, [x_train.shape[0], 28 * 28])
    flat_test = np.reshape(x_test, [x_test.shape[0], 28 * 28])

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(28 * 28), dtype='float32'))
    model.add(Dense(2000, kernel_regularizer=tf.keras.regularizers.l1(l1=config["l1"]), use_bias=False, activation='tanh'))
    model.add(Dense(10, use_bias=False, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=config["lr"])

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    epochs = config["epochs"]
    model.fit(flat_train, y_train, batch_size=500, epochs=epochs, verbose=0)
    testacc = model.evaluate(flat_test, y_test, batch_size=700, verbose=0)
    tune.report(acc=testacc[1])

result = tune.run(train_funct,
                  num_samples=1,
                  metric="acc",
                  mode="max",
                  resources_per_trial=trialResources,
                  config=config)

best_trial = result.get_best_trial("acc", "max", "avg")
print("Best trial config: {}".format(best_trial.config))