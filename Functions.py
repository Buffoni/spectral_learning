"""
Functions used to examine layer's structure after the training and compare the 'Dense' layer and the 'Spectral' one.
The 'model_config' file contains the structure of the network to be tested.
The dataset loaded is MNIST
@authors: Lorenzo Buffoni, Lorenzo Giambagli
"""

import tensorflow as tf
from tqdm import tqdm
from SpectralLayer import Spectral
from tensorflow.keras.layers import Dense
from tensorflow.keras.backend import set_value
import numpy as np
from numpy import multiply as mult
import pickle as pk

# Parallel execution's stuff
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_synchronous_execution(False)

dataset = tf.keras.datasets.fashion_mnist

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

def train_model(config):
    """
    Train a configurated model according to model_config
    :return: Opens a file in append mode and write down the Test_accuracy
    """
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
    # percentiles = list(range(0, 105, 5))
    percentiles = list(np.arange(1, 100, 5))
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


def dense_connectivity_trim_SL(model):
    f = open('testset.pickle', 'rb')
    x_test, y_test = pk.load(f)
    f.close()
    percentiles = list(np.arange(1, 100, 5))
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

def dense_pruning_SL(model):
    f = open('testset.pickle', 'rb')
    x_test, y_test = pk.load(f)
    f.close()
    percentiles = list(np.arange(1, 100, 5))


def val_vec_train_trim(config):
    model_config = config

    mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

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

    percentiles = list(np.arange(1, 100, 5))
    results = {"percentile": [], "val_accuracy": []}
    diag = model.layers[0].diag.numpy()
    abs_diag = np.abs(diag)
    thresholds = [np.percentile(abs_diag, q=perc) for perc in percentiles]

    model_config["is_base"] = [True]
    model_config["is_diag"] = [True]
    model_config["regularize"] = [None]

    for t, perc in tqdm(list(zip(thresholds, percentiles)), desc="  Removing the eigenvalues and train vectors"):

        diag[abs_diag < t] = 0.0
        hid_size = np.count_nonzero(diag)

        #Smaller Model
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

        new_model.layers[0].diag.assign(diag[diag != 0.0])
        tmp_base = model.layers[0].base.numpy()
        new_model.layers[0].base.assign(tmp_base[:, diag != 0.0])

        new_model.layers[1].diag.assign(model.layers[1].diag)
        tmp_base = model.layers[1].base.numpy()
        new_model.layers[1].base.assign(tmp_base[diag != 0.0, :])

        new_model.fit(flat_train, y_train, verbose=0, batch_size=model_config['batch_size'], epochs=model_config['epochs'])
        test_results = new_model.evaluate(flat_test, y_test, batch_size=800, verbose=0)
        results["percentile"].append(perc)
        results["val_accuracy"].append(test_results[1])

    return [results["percentile"], results["val_accuracy"]]

def autoval_vec_train_test(config):
    global model_config
    model_config = config

    mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    flat_train = np.reshape(x_train, [x_train.shape[0], 28 * 28])
    flat_test = np.reshape(x_test, [x_test.shape[0], 28 * 28])

    file = open('testset.pickle', 'wb')
    tmp = [flat_test, y_test]
    pk.dump(tmp, file)
    file.close()

    model_config["is_base"] = [False]
    model_config["last_is_base"] = True
    model = build_feedforward(model_config)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'], run_eagerly=False)

    model.fit(flat_train, y_train, verbose=0, batch_size=model_config['batch_size'], epochs=model_config['epochs'])
    print("\nStart acc:\n")
    model.evaluate(flat_test, y_test, batch_size=500, verbose=2)

    diag = model.layers[0].diag.numpy()
    abs_diag = np.abs(diag)
    thresholds = np.percentile(abs_diag, q=96)

    diag[abs_diag < thresholds] = 0.0
    print("Nodi rimasti: {}\n".format(np.count_nonzero(diag)))
    model.layers[0].diag.assign(diag)

    print("\nPost Taglio:\n")
    model.evaluate(flat_test, y_test, batch_size=500, verbose=2)

    model_config["is_base"] = [True]
    model_config["is_diag"] = [False]
    model_config["regularize"] = [None]

    new_model = build_feedforward(model_config)

    for i in range(2):
        new_model.layers[i].diag.assign(model.layers[i].diag)
        new_model.layers[i].base.assign(model.layers[i].base)

    print("\nAccuracy post-assegnazione:\n")
    new_model.evaluate(flat_test, y_test, batch_size=300, verbose=1)

    new_model.fit(flat_train, y_train, verbose=0, batch_size=model_config['batch_size'], epochs=model_config['epochs'])
    print("\nAccuracy finale:\n")
    new_model.evaluate(flat_test, y_test, batch_size=300, verbose=1)

def spectral_eigval_trim(model, cut_n=80):
    """
    :param model: spectral model to Trim
    :param cut_n: numbers of steps between minimum and maximum of the eigenvalues
    :return: accuracy per cut and ratio between eigenvalues used and the total number of eigenvalues
    """

    f = open('testset.pickle', 'rb')
    flat_test, y_test = pk.load(f)
    f.close()

    cut_min = np.zeros([len(model.layers) - 1])
    cut_max = np.zeros([len(model.layers) - 1])
    cut = []
    eig_n = np.array(model_config['size']).sum()


    for i in range(0, len(model.layers) - 1):
        cut_max[i] = abs(model.layers[i].get_weights()[1]).max()
        cut_min[i] = abs(model.layers[i].get_weights()[1]).min()
        cut_step = (cut_max[i] - cut_min[i]) / cut_n
        cut.append(np.arange(cut_min[i], cut_max[i], cut_step).tolist())

    eig_ratio = []
    acc_final = []

    for j in range(0, cut_n):
        useful_eig = 0
        for i in range(0, len(model.layers) - 1):
            diag_out = model.layers[i].get_weights()[1]
            diag_out[abs(diag_out) < cut[i][j]] = 0
            set_value(model.layers[i].diag, tf.constant(diag_out, dtype=tf.float32))
            useful_eig = useful_eig + np.count_nonzero(diag_out != 0)


        model.compile(optimizer=model.optimizer,
                      loss=model.loss,
                      metrics=['accuracy'],
                      run_eagerly=True)

        tested = model.evaluate(flat_test, y_test, batch_size=1000, verbose=0)
        acc_final.append(tested[1])
        eig_ratio.append(useful_eig / eig_n)

    return np.array(eig_ratio), np.array(acc_final)

def dense_connectivity_trim(model, cut_n=80):
    """
    :param model: Dense model to Trim using incoming connectivity
    :param flat_test: testset elements
    :param y_test: testset labels
    :param cut_n: numbers of steps between minimum and maximum of the eigenvalues
    :return: accuracy per cut and ratio between nodes used and the total number of nodes
    """
    f = open('testset.pickle', 'rb')
    flat_test, y_test = pk.load(f)
    f.close()

    nodes_n = np.array(model_config['size']).sum()

    cut_min = np.zeros([len(model.layers) - 1])
    cut_max = np.zeros([len(model.layers) - 1])
    cut = []

    #Range of the connectivity
    for i in range(0, len(model.layers) - 1):
        pesi = model.layers[i].get_weights()[0]
        connectivity = np.sum(pesi, axis=0)
        cut_max[i] = abs(connectivity).max()
        cut_min[i] = abs(connectivity).min()

        cut_step = (cut_max[i] - cut_min[i]) / cut_n
        cut.append(np.arange(cut_min[i], cut_max[i], cut_step).tolist())

    nodes_ratio = []
    acc_final = []

    for j in range(0, cut_n):
        nonzero = 0
        for i in range(0, len(model.layers) - 1):
            pesi = model.layers[i].get_weights()[0]
            pesi = pesi.T
            connectivity = np.sum(pesi, axis=1)
            filtro = abs(connectivity) > cut[i][j]
            filtro.astype(np.int)
            new = pesi.T * filtro
            model.layers[i].set_weights([new])
            nonzero = nonzero + filtro.sum()

        model.compile(optimizer=model.optimizer,
                      loss=model.loss,
                      metrics=['accuracy'],
                      run_eagerly=False)

        tested = model.evaluate(flat_test, y_test, batch_size=1000, verbose=0)
        acc_final.append(tested[1])
        nodes_ratio.append(nonzero / nodes_n)

    return np.array(nodes_ratio), np.array(acc_final)

def spectral_connectivity_trim(model):
    dense_eq = tf.keras.Sequential()
    dense_eq.add(tf.keras.layers.Input(shape=model_config['input_shape'], dtype='float32'))

    for i in range(0, len(model.layers) - 1):
        dense_eq.add(Dense(model_config['size'][i], use_bias=model_config['is_bias'][i],
                                           activation=model_config['activ'][i]))

    dense_eq.add(Dense(model_config['last_size'], use_bias=model_config['last_is_bias'],
                                       activation=model_config['last_activ']))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    dense_eq.compile(optimizer=opt,
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'], run_eagerly=False)

    for i in range(0, len(model.layers)): #Spectral layers
        print(i)
        diag = model.layers[i].get_weights()[1]
        base = model.layers[i].get_weights()[0]
        w = - mult(base, diag)
        dense_eq.layers[i].set_weights([w])

    return dense_connectivity_trim(dense_eq)

def mod_connectivity_trim(model, cut_n=80):
    f = open('testset.pickle', 'rb')
    flat_test, y_test = pk.load(f)
    f.close()

    nodes_n = np.array(model_config['size']).sum()

    cut_min = np.zeros([len(model.layers) - 1])
    cut_max = np.zeros([len(model.layers) - 1])
    cut = []

    #Range of the connectivity
    for i in range(0, len(model.layers) - 1):
        pesi = model.layers[i].get_weights()[0]
        m_connectivity = np.sum(abs(pesi), axis=0)
        cut_max[i] = m_connectivity.max()
        cut_min[i] = m_connectivity.min()

        cut_step = (cut_max[i] - cut_min[i]) / cut_n
        cut.append(np.arange(cut_min[i], cut_max[i], cut_step).tolist())

    nodes_ratio = []
    acc_final = []

    for j in range(0, cut_n):
        nonzero = 0
        for i in range(0, len(model.layers) - 1):
            pesi = model.layers[i].get_weights()[0]
            pesi = pesi.T
            m_connectivity = np.sum(abs(pesi), axis=1)
            filtro = m_connectivity > cut[i][j]
            filtro.astype(np.int)
            new = pesi.T * filtro
            model.layers[i].set_weights([new])
            nonzero = nonzero + filtro.sum()

        model.compile(optimizer=model.optimizer,
                      loss=model.loss,
                      metrics=['accuracy'])

        tested = model.evaluate(flat_test, y_test, batch_size=1000, verbose=0)
        acc_final.append(tested[1])
        nodes_ratio.append(nonzero / nodes_n)

    return np.array(nodes_ratio), np.array(acc_final)

