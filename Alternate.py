from Function_Alternate import *
import matplotlib.pyplot as plt
import seaborn as sb
import pickle as pk

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
    'dense_regularize' : [None],
    'is_bias': [False],  # True means a trainable bias, False ow
    'activ': ['relu'],  # Activation function

    # Same parameters but for the last layer
    'last_type': 'spec',
    'last_activ': 'softmax',
    'last_size': 10,
    'last_is_base': True,
    'last_is_diag': True,
    'last_is_bias': False,

    # Training Parameters
    'batch_size': 300,
    'epochs': 100,
    'normalize': False
}

plt.figure(0, dpi=200)
Results = {"lay": [], "percentile": [], "val_accuracy": []}

N = 2
for i in range(N):
    print('Spectral Val/Vec...\n')
    model_config['is_base'] = [False]
    [x, y] = spectral_alternate(config=model_config)
    Results["lay"].extend(['Spectral_Alt.'] * len(x))
    Results["percentile"].extend(x)
    Results["val_accuracy"].extend(y)

for i in range(N):
    print('Dense AbsConn...\n')
    model_config['type'] = ['dense']
    model_config['last_type'] = ['dense']
    [x, y] = dense_alternate(config=model_config)
    Results["lay"].extend(['Dense_Alt.'] * len(x))
    Results["percentile"].extend(x)
    Results["val_accuracy"].extend(y)




accuracy_perc_plot = sb.lineplot(x="percentile", y="val_accuracy", hue="lay", style="lay",
                                 markers=True, dashes=False, ci="sd", data=Results)

# accuracy_perc_plot.get_figure().savefig("./Fmnist/Fmnist_relu.png")
plt.show()

f = open("./backup/mnist_relu_N.p", "wb")
pk.dump(Results, f)

