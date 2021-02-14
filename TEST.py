from Function_Alternate import *
import matplotlib.pyplot as plt
import seaborn as sb
import pickle as pk

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_synchronous_execution(False)

model_config = {
    'input_shape': 784,
    'type': ['dense'],  # Types of hidden layers: 'spec' = Spectral Layer, second diag trainable, 'dense' = Dense layer
    'size': [800],  # Size of every hidden layer
    'is_base': [True],  # True means a trainable basis, False ow
    'is_diag': [True],  # True means a trainable eigenvalues, False ow
    'regularize': [None],
    'dense_regularize' : [None],
    'is_bias': [False],  # True means a trainable bias, False ow
    'activ': ['tanh'],  # Activation function

    # Same parameters but for the last layer
    'last_type': 'dense',
    'last_activ': 'softmax',
    'last_size': 10,
    'last_is_base': True,
    'last_is_diag': True,
    'last_is_bias': False,

    # Training Parameters
    'batch_size': 500,
    'epochs': 30,

}

model = build_feedforward(model_config)
