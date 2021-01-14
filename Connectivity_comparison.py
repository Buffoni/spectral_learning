from Functions import *
import matplotlib.pyplot as plt
from collections import OrderedDict

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
    'last_is_base': True,
    'last_is_diag': True,
    'last_is_bias': False,

    # Training Parameters
    'batch_size': 800,
    'epochs': 20
}

plt.figure(0)
for i in range(5):
    print(f"Trial: {i + 1}\n")

    print('Dense...\n')
    model_config['type'] = ['dense']
    model_config['last_type'] = 'dense'
    dense_mod = train_model(config=model_config)
    dense_copy = dense_mod
    [dense_conn, dense_acc_conn] = dense_connectivity_trim(dense_mod)
    [dense_conn_mod, dense_acc_conn_mod] = mod_connectivity_trim(dense_mod)

    plt.plot(dense_conn_mod, dense_acc_conn_mod, 'ro', markersize=2, label='Spec-Conn')
    plt.plot(dense_conn, dense_acc_conn, 'bo', markersize=2, label='Dense-Conn')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.title('tanh-1 hid-connect_comparison')
plt.xlabel('Active Nodes Fraction', fontsize=15)
plt.ylabel('Test accuracy', fontsize=15)
plt.savefig("Figure/Connect-comp.png")
plt.show()
