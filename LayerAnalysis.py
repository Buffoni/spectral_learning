"""
Functions used to examine layer's structure after the training and compare the 'Dense' layer and the 'Spectral' one.
The 'model_config' file contains the structure of the network to be tested.
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

model_config = {
    'input_shape': 784,
    'type': ['spec'],  # Types of hidden layers: 'spec' = Spectral Layer, second diag trainable, 'dense' = Dense layer
    'size': [2000],  # Size of every hidden layer
    'is_base': [True],  # True means a trainable basis, False ow
    'is_diag': [True],  # True means a trainable eigenvalues, False ow
    'regularize': [None],
    'is_bias': [False],  # True means a trainable bias, False ow
    'activ': ['tanh'],  # Activation function

    # Same parameters but for the last layer
    'last_type': 'spec',
    'last_activ': 'softmax',
    'last_size': 10,
    'last_diag_portion': '2',
    'last_is_base': True,
    'last_is_diag': True,
    'last_is_bias': False,

    # Training Parameters
    'batch_size': 800,
    'epochs': 30
}


def build_feedforward():
    """
    :param config: Configuration file for your model
    :return: Model created according to 'model_config'
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=model_config['input_shape'], dtype='float32'))

    for i in range(0, len(model_config['type'])):
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
                            activation=model_config['activ'][i]))

    if model_config['last_type'] == 'spec':
        model.add(Spectral(model_config['last_size'],
                           is_diag_trainable=model_config['last_is_diag'],
                           is_base_trainable=model_config['last_is_base'],
                           is_bias=model_config['last_is_bias'],
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

def train_model(config, load_model=False):
    """
    :return: Opens a file in append mode and write down the Test_accuracy
    """
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

    model.fit(flat_train, y_train, verbose=0, batch_size=model_config['batch_size'], epochs=config['epochs'])
    model.evaluate(flat_test, y_test, batch_size=1000, verbose=1)
    return model

def lambda_trimmer(model, cut_n=60):
    """
    :param model: spectral model to Trim
    :param cut_n: numbers of steps between minimum and maximum of the eigenvalues
    :return: accuracy per cut and ratio between eigenvalues used and the total number of eigenvalues
    """
    f = open('testset.pickle', 'rb')
    flat_test, y_test = pk.load(f)
    f.close()

    autov_coll = np.array([])
    for i in range(0, len(model.layers) - 1):
        autov_coll = np.concatenate((autov_coll, model.layers[i].get_weights()[1].numpy()), axis=0)
    eig_n = len(autov_coll)
    print(eig_n)

    cut_min = abs(autov_coll).min()
    cut_max = abs(autov_coll).max()
    cut_step = (cut_max - cut_min) / cut_n
    cut = np.arange(cut_min, cut_max, cut_step).tolist()

    eig_ratio = []
    acc_final = []
    for c in cut:
        useful_eig = 0
        for j in range(0, len(model.layers) - 1):
            diag_out = model.layers[j].get_weights()[1]
            diag_out[abs(diag_out) < c] = 0
            set_value(model.layers[j].diag, tf.constant(diag_out, dtype=tf.float32))
            useful_eig = useful_eig + np.count_nonzero(diag_out != 0)

        model.compile(optimizer=model.optimizer,
                      loss=model.loss,
                      metrics=['accuracy'],
                      run_eagerly=False)

        tested = model.evaluate(flat_test, y_test, batch_size=1000, verbose=0)
        acc_final.append(tested[1])
        eig_ratio.append(useful_eig / eig_n)

    return np.array(acc_final), np.array(eig_ratio)


def connectivity_trimmer(model, cut_n=60):
    """
    :param model: dense model to Trim
    :param flat_test: testset elements
    :param y_test: testset labels
    :param cut_n: numbers of steps between minimum and maximum of the eigenvalues
    :return: accuracy per cut and ratio between nodes used and the total number of nodes
    """
    f = open('testset.pickle', 'rb')
    flat_test, y_test = pk.load(f)
    f.close()
    connect_coll = np.array([])

    for i in range(0, len(model.layers) - 1):
        pesi = model.layers[i].get_weights()[0]
        pesi = pesi.T
        connectivity = np.sum(pesi, axis=1)
        connect_coll = np.concatenate((connect_coll, connectivity), axis=0)

    nodes_n = len(connect_coll)
    print(nodes_n)
    cut_min = abs(connect_coll).min()
    cut_max = abs(connect_coll).max()
    cut_step = (cut_max - cut_min) / cut_n
    cut = np.arange(cut_min, cut_max, cut_step).tolist()
    nodes_ratio = []
    acc_final = []

    for c in cut:
        nonzero = 0
        for j in range(0, len(model.layers) - 1):
            pesi = model.layers[j].get_weights()[0]
            pesi = pesi.T
            connectivity = np.sum(pesi, axis=1)
            filtro = abs(connectivity) > c
            filtro.astype(np.int)
            new = pesi.T * filtro
            model.layers[j].set_weights([new])
            nonzero = nonzero + filtro.sum()

        model.compile(optimizer=model.optimizer, loss=model.loss, metrics=['accuracy'], run_eagerly=False)
        tested = model.evaluate(flat_test, y_test, batch_size=1000, verbose=0)
        acc_final.append(tested[1])
        nodes_ratio.append(nonzero / nodes_n)

    return np.array(acc_final), np.array(nodes_ratio)

def dense_equiv_trimmer(model, cut_n=60):
    dense_eq = tf.keras.Sequential()
    dense_eq.add(tf.keras.layers.Input(shape=model_config['input_shape'], dtype='float32'))
    for i in range(0, len(model.layers) - 1):
        dense_eq.add(tf.keras.layers.Dense(model_config['size'][i], use_bias=model_config['is_bias'][i],
                                           activation=model_config['activ'][i]))

    dense_eq.add(tf.keras.layers.Dense(model_config['last_size'], use_bias=model_config['last_is_bias'],
                                       activation=model_config['last_activ']))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    dense_eq.compile(optimizer=opt,
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'], run_eagerly=False)

    for i in range(0, len(dense_eq.layers)):
        diag = np.diagflat(model.layers[i].get_weights()[1])
        base = model.layers[i].get_weights()[0]
        w = - mult(base, diag)
        dense_eq.layers[i].set_weights([w])

    return connectivity_trimmer(dense_eq)


def phi_and_lambda(config):
    import pickle as pk
    f = open(r'C:\Users\loren\PycharmProjects\Trimming\base.p', "wb")
    model_config['is_base'] = [True]
    model_config['diag_portion'] = ['no']
    trained_model = train_model(config, load_model=False)
    model_config['is_base'] = [False]
    model_config['diag_portion'] = ['2']
    bases = []
    for l in range(0, len(trained_model.layers)):
        bases.append(trained_model.layers[l].base.numpy())
    pk.dump(bases, f)
    f.close()

    o = open(r'C:\Users\loren\PycharmProjects\Trimming\diags.p', "wb")
    diagon = []
    for l in range(0, len(trained_model.layers)):
        diagon.append(trained_model.layers[l].eigval_2.numpy())

    pk.dump(diagon, o)
    o.close()
    print('Retrain\n')
    trained2_model = train_model(config, load_model=True)
    FL = open(r'C:\Users\loren\PycharmProjects\Trimming\Risultati.p', "ab")
    return lambda_trimmer(trained2_model)
    # to_dump = [ris_trim1, ris_trim2]
    # pk.dump(to_dump, FL)
    # FL.close()
    # print('\nTrimmed!\n')





def simultaneus_train(config):
    import pickle as pk

    model_config['type'] = ['spec']
    model_config['last_type'] = 'spec'

    spec_full = train_model(config)
    spec_copy = spec_full

    g = open(r'C:\Users\loren\PycharmProjects\Trimming\Data\Spec_connectiv.p', "ab")
    print('Layer Spectral, Trim Connectivity\n')
    acc1, ratio1 = dense_equiv_trimmer(spec_full)
    ris1 = [acc1, ratio1]
    pk.dump(ris1, g)

    f = open(r'C:\Users\loren\PycharmProjects\Trimming\Data\Simultaneus_spec.p', "ab")
    print('Layer Spectral, Trim Lambda\n')
    acc, ratio = lambda_trimmer(spec_copy)
    ris = [acc, ratio]
    pk.dump(ris, f)

    model_config['type'] = ['dense']
    model_config['last_type'] = 'dense'
    dense = train_model(config)
    f = open(r'C:\Users\loren\PycharmProjects\Trimming\Data\Connectivity.p', "ab")
    print('Layer Dense, Trim Connectivity\n')
    acc, ratio = connectivity_trimmer(dense)
    ris = [acc, ratio]
    pk.dump(ris, f)
    return None
