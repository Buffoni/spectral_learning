"""
Functions used to examine layer's structure after the training and compare the 'Dense' layer and the 'Spectral' one.
The 'model_config' file contains the structure of the network to be tested.
The dataset loaded is MNIST
@authors: Lorenzo Buffoni, Lorenzo Giambagli
"""
import tensorflow as tf
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

# model_config = {
#     'input_shape': 784,
#     'type': ['spec'],    # Types of hidden layers: 'spec' = Spectral Layer, second diag trainable, 'dense' = Dense layer
#     'size': [2000, 2000],      # Size of every hidden layer
#     'is_base': [True, True],   # True means a trainable basis, False ow
#     'is_diag': [True, True],   # True means a trainable eigenvalues, False ow
#     'regularize': [None, None],
#     'is_bias': [False, False],  # True means a trainable bias, False ow
#     'activ': ['tanh', 'tanh'],   # Activation function
#
#     # Same parameters but for the last layer
#     'last_type': 'spec',
#     'last_activ': 'softmax',
#     'last_size': 10,
#     'last_is_base': True,
#     'last_is_diag': True,
#     'last_is_bias': False,
#
#     # Training Parameters
#     'batch_size': 800,
#     'epochs': 20
# }

model_config = None

def build_feedforward():
    """
    :param config: Configuration file for your model
    :return: Model created according to 'model_config'
    """

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=model_config['input_shape'], dtype='float32'))

    for i in range(0, len(model_config['size'])-1):
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
                            kernel_regularizer=model_config['regularize'][i],
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

def train_model(config=None, load_model=False):
    """
    Train a configurated model according to model_config
    :return: Opens a file in append mode and write down the Test_accuracy
    """
    # if config is not None:
    model_config = config

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    flat_train = np.reshape(x_train, [x_train.shape[0], 28 * 28])
    flat_test = np.reshape(x_test, [x_test.shape[0], 28 * 28])

    file = open('testset.pickle', 'wb')
    tmp = [flat_test, y_test]
    pk.dump(tmp, file)
    file.close()

    if load_model:
        print('Loading from previous model...\n')
        f = open(r'C:\Users\loren\PycharmProjects\Trimming\base.p', "rb")
        while True:
            try:
                to_load_phi = pk.load(f)
            except EOFError:
                break

        f = open(r'C:\Users\loren\PycharmProjects\Trimming\diags.p', "rb")
        while True:
            try:
                to_load_diag = pk.load(f)
            except EOFError:
                break

        model = build_feedforward()
        model.compile(optimizer=model.optimizer,
                      loss=model.loss,
                      metrics=['accuracy'],
                      run_eagerly=False)

        for l in range(0, len(model.layers)):
            model.layers[l].base = tf.constant(to_load_phi[l], dtype=tf.float32)
            model.layers[l].eigval_2 = tf.Variable(to_load_diag[l], dtype=tf.float32)

            model.compile(optimizer=model.optimizer,
                          loss=model.loss,
                          metrics=['accuracy'],
                          run_eagerly=False)
    else:
        model = build_feedforward()
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'], run_eagerly=False)

    model.fit(flat_train, y_train, verbose=0, batch_size=model_config['batch_size'], epochs=model_config['epochs'])
    model.evaluate(flat_test, y_test, batch_size=1000, verbose=0)

    return model

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
    eig_n = np.array(model_config['size']).sum() / 2

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
        # for i in range(0, len(model.layers) - 2):
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

    nodes_n = np.array(model_config['size']).sum() /2

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
        # for i in range(0, len(model.layers) - 1):
        for i in range(0, len(model.layers) - 2):
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
        diag = model.layers[i].get_weights()[1]
        base = model.layers[i].get_weights()[0]
        w = - mult(base, diag)
        dense_eq.layers[i].set_weights([w])

    return dense_connectivity_trim(dense_eq)

