"""
Functions used to examine layer's structure after the training and compare the 'Dense' layer and the 'Spectral' one.
The 'model_config' file contains the structure of the network to be tested.
The dataset loaded is MNIST
@authors: Lorenzo Buffoni, Lorenzo Giambagli
"""
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tqdm import tqdm
from SpectralLayer import Spectral
from tensorflow.keras.layers import Dense
from tensorflow.keras.backend import set_value
import numpy as np
from numpy import multiply as mult
import pickle as pk

# Parallel execution stuff
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_synchronous_execution(False)

dataset = tf.keras.datasets.fashion_mnist
perc_span = np.arange(97, 100, 0.2)

def build_feedforward(model_config):
    """
    :param config: Configuration file for your model
    :return: Model created according to 'model_config'
    """

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=model_config['input_shape'], dtype='float32'))

    for i in range(0, len(model_config['size'])):
        if model_config['type'][i] == 'spec':
            model.add(Spectral(model_config['size'][i],
                               is_diag_trainable=model_config['is_diag'][i],
                               is_base_trainable=model_config['is_base'][i],
                               diag_regularizer=model_config['regularize'][i],
                               use_bias=model_config['is_bias'][i],
                               activation=model_config['activ'][i]))
        else:
            model.add(Dense(model_config['size'][i],
                            use_bias=model_config['is_bias'][i],
                            kernel_regularizer=model_config['dense_regularize'][i],
                            activation=model_config['activ'][i]))

    if model_config['last_type'] == 'spec':
        model.add(Spectral(model_config['last_size'],
                               is_diag_trainable=model_config['last_is_diag'],
                               is_base_trainable=model_config['last_is_base'],
                               use_bias=model_config['last_is_bias'],
                               activation=model_config['last_activ']))
    else:
        model.add(Dense(model_config['last_size'],
                        use_bias=model_config['last_is_bias'],
                        activation=model_config['last_activ']))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'], run_eagerly=False)
    return model

def train_model(config, normalize = False):
    """
    Train a configurated model according to model_config
    :return: Opens a file in append mode and write down the Test_accuracy
    """
    model_config = config

    mnist = dataset

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    normalize = config['normalize']

    if normalize:
        x_train, x_test = x_train-np.mean(x_train), x_test-np.mean(x_test)

    flat_train = np.reshape(x_train, [x_train.shape[0], 28 * 28])
    flat_test = np.reshape(x_test, [x_test.shape[0], 28 * 28])

    file = open('testset.pickle', 'wb')
    tmp = [flat_test, y_test]
    pk.dump(tmp, file)
    file.close()

    model = build_feedforward(model_config)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'], run_eagerly=False)

    model.fit(flat_train, y_train, verbose=0, batch_size=model_config['batch_size'], epochs=model_config['epochs'])

    return model

def spectral_eigval_trim_SL(model):

    f = open('testset.pickle', 'rb')
    x_test, y_test = pk.load(f)
    f.close()
    percentiles = list(perc_span)
    results = {"diag_reg": [], "percentile": [], "val_accuracy": []}
    diag = model.layers[0].diag.numpy()
    abs_diag = np.abs(diag)
    thresholds = [np.percentile(abs_diag, q=perc) for perc in percentiles]
    for t, perc in tqdm(list(zip(thresholds, percentiles)), desc="  Removing the eigenvalues"):
        diag[abs_diag < t] = 0.0
        model.layers[0].diag.assign(diag)
        test_results = model.evaluate(x_test, y_test, batch_size=1000, verbose=0)
        # storing the results
        results["percentile"].append(perc)
        results["val_accuracy"].append(test_results[1])

    return [results["percentile"], results["val_accuracy"]]

def spectral_encoder_SL(model):

    f = open('testset.pickle', 'rb')
    x_test, y_test = pk.load(f)
    x_test = x_test.reshape([x_test.shape[0], 28, 28])
    f.close()
    percentiles = list(perc_span)
    results = {"percentile": [], "val_loss": []}
    diag = model.encoder.layers[1].diag.numpy()
    abs_diag = np.abs(diag)
    thresholds = [np.percentile(abs_diag, q=perc) for perc in percentiles]
    for t, perc in tqdm(list(zip(thresholds, percentiles)), desc="  Removing the eigenvalues"):
        diag[abs_diag < t] = 0.0
        model.encoder.layers[1].diag.assign(diag)
        test_results = model.evaluate(x_test, x_test, batch_size=1000, verbose=0)
        # storing the results
        results["percentile"].append(perc)
        results["val_loss"].append(test_results)

    return [results["percentile"], results["val_loss"]]

def dense_connectivity_trim_SL(model):
    f = open('testset.pickle', 'rb')
    x_test, y_test = pk.load(f)
    f.close()
    percentiles = list(perc_span)
    # percentiles = list(np.arange(97, 100, 0.1))
    weights = model.layers[0].weights[0].numpy()
    connectivity = np.abs(weights).sum(axis=0)
    thresholds = [np.percentile(connectivity, q=perc) for perc in percentiles]
    results = {"percentile": [], "val_accuracy": []}

    for t, perc in tqdm(list(zip(thresholds, percentiles)), desc="  Removing the nodes"):
        weights[:, connectivity < t] = 0.0
        model.layers[0].weights[0].assign(weights)
        test_results = model.evaluate(x_test, y_test, batch_size=1000, verbose=0)
        # storing the results
        results["percentile"].append(perc)
        results["val_accuracy"].append(test_results[1])

    return [results["percentile"], results["val_accuracy"]]

def dense_encoder_trim_SL(model):
    f = open('testset.pickle', 'rb')
    x_test, y_test = pk.load(f)
    x_test = x_test.reshape([x_test.shape[0], 28, 28])
    f.close()
    percentiles = list(perc_span)
    # percentiles = list(np.arange(97, 100, 0.1))
    weights = model.encoder.layers[1].weights[0].numpy()
    connectivity = np.abs(weights).sum(axis=0)
    thresholds = [np.percentile(connectivity, q=perc) for perc in percentiles]
    results = {"percentile": [], "val_loss": []}

    for t, perc in tqdm(list(zip(thresholds, percentiles)), desc="  Removing the nodes"):
        weights[:, connectivity < t] = 0.0
        model.encoder.layers[1].weights[0].assign(weights)
        test_results = model.evaluate(x_test, x_test, batch_size=1000, verbose=0)
        # storing the results
        results["percentile"].append(perc)
        results["val_loss"].append(test_results)

    return [results["percentile"], results["val_loss"]]

def spectral_alternate(config):
    model_config = config

    mnist = dataset

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    flat_train = np.reshape(x_train, [x_train.shape[0], 28 * 28])
    flat_test = np.reshape(x_test, [x_test.shape[0], 28 * 28])

    file = open('testset.pickle', 'wb')
    tmp = [flat_test, y_test]
    pk.dump(tmp, file)
    file.close()

    #Initial Spectral Net
    model = build_feedforward(model_config)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'], run_eagerly=False)

    model.fit(flat_train, y_train, verbose=0, batch_size=model_config['batch_size'], epochs=model_config['epochs'])

    percentiles = list(perc_span)
    results = {"percentile": [], "val_accuracy": []}
    diag = model.layers[0].diag.numpy()
    abs_diag = np.abs(diag)
    thresholds = [np.percentile(abs_diag, q=perc) for perc in percentiles]

    model_config["is_base"] = [True]
    model_config["is_diag"] = [True]
    model_config["regularize"] = [None]

    for t, perc in tqdm(list(zip(thresholds, percentiles)), desc="Removing the eigenvalues and train vectors"):

        diag[abs_diag < t] = 0.0
        hid_size = np.count_nonzero(diag)

        #Smaller Model after Trimming
        new_model = tf.keras.Sequential()
        new_model.add(tf.keras.layers.Input(shape=model_config['input_shape'], dtype='float32'))

        i = 0
        new_model.add(Spectral(hid_size,
                           is_diag_trainable=model_config['is_diag'][i],
                           is_base_trainable=model_config['is_base'][i],
                           diag_regularizer=model_config['regularize'][i],
                           use_bias=model_config['is_bias'][i],
                           activation=model_config['activ'][i]))

        new_model.add(Spectral(model_config['last_size'],
                           is_diag_trainable=model_config['last_is_diag'],
                           is_base_trainable=model_config['last_is_base'],
                           use_bias=model_config['last_is_bias'],
                           activation=model_config['last_activ']))

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        new_model.compile(optimizer=opt,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'], run_eagerly=False)

        #Setting the smaller network
        new_model.layers[0].diag.assign(diag[diag != 0.0])
        tmp_base = model.layers[0].base.numpy()
        new_model.layers[0].base.assign(tmp_base[:, diag != 0.0])

        new_model.layers[1].diag.assign(model.layers[1].diag)
        tmp_base = model.layers[1].base.numpy()
        new_model.layers[1].base.assign(tmp_base[diag != 0.0, :])
        #Train smaller network
        new_model.fit(flat_train, y_train, verbose=0, batch_size=model_config['batch_size'], epochs=30)

        test_results = new_model.evaluate(flat_test, y_test, batch_size=1000, verbose=0)
        results["percentile"].append(perc)
        results["val_accuracy"].append(test_results[1])

    return [results["percentile"], results["val_accuracy"]]

def dense_alternate(config):
    model_config = config

    mnist = dataset

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    flat_train = np.reshape(x_train, [x_train.shape[0], 28 * 28])
    flat_test = np.reshape(x_test, [x_test.shape[0], 28 * 28])

    file = open('testset.pickle', 'wb')
    tmp = [flat_test, y_test]
    pk.dump(tmp, file)
    file.close()

    # Initial Dense Net
    model = build_feedforward(model_config)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'], run_eagerly=False)

    model.fit(flat_train, y_train, verbose=0, batch_size=model_config['batch_size'], epochs=model_config['epochs'])

    percentiles = list(perc_span)
    results = {"percentile": [], "val_accuracy": []}

    w0 = model.layers[0].get_weights()[0]
    w1 = model.layers[1].get_weights()[0]

    abs_conn = abs(w0).sum(axis=0)

    thresholds = [np.percentile(abs_conn, q=perc) for perc in percentiles]

    for t, perc in tqdm(list(zip(thresholds, percentiles)), desc="Removing nodes"):
        abs_conn[abs_conn < t] = 0.0
        hid_size = np.count_nonzero(abs_conn)

        # Smaller Model after Trimming
        new_model = tf.keras.Sequential()
        new_model.add(tf.keras.layers.Input(shape=model_config['input_shape'], dtype='float32'))

        i = 0
        new_model.add(Dense(hid_size,
                               kernel_regularizer=model_config['regularize'][i],
                               use_bias=model_config['is_bias'][i],
                               activation=model_config['activ'][i]))

        new_model.add(Dense(model_config['last_size'],
                               use_bias=model_config['last_is_bias'],
                               activation=model_config['last_activ']))

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        new_model.compile(optimizer=opt,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'], run_eagerly=False)

        # Setting the smaller network
        new_model.layers[0].set_weights([w0[:, abs_conn >= t], ])

        new_model.layers[1].set_weights([w1[abs_conn >= t, :], ])

        # Train smaller network
        new_model.fit(flat_train, y_train, verbose=0, batch_size=model_config['batch_size'], epochs=30)

        test_results = new_model.evaluate(flat_test, y_test, batch_size=1000, verbose=0)
        results["percentile"].append(perc)
        results["val_accuracy"].append(test_results[1])

    return [results["percentile"], results["val_accuracy"]]
