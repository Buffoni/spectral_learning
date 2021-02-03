from Functions import *
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
    'regularize': ['l1'],
    'dense_regularize' : [tf.keras.regularizers.l1(l1=0.0005)],
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
    'batch_size': 200,
    'epochs': 50
}

plt.figure(0, dpi=200)

Results = {"lay": [], "percentile": [], "val_accuracy": []}

N = 1
for i in range(N):
    print(f"Trial: {i + 1}\n")

    print('Spectral...\n')
    model_config['type'] = ['spec']
    model_config['is_base'] = [True]
    model_config['last_type'] = 'spec'
    spec_mod = train_model(config=model_config)
    [x, y] = spectral_eigval_trim_SL(spec_mod)
    Results["lay"].extend(['spec'] * len(x))
    Results["percentile"].extend(x)
    Results["val_accuracy"].extend(y)
# for i in range(N):
#     print('Dense Retrain...\n')
#     model_config['type'] = ['dense']
#     model_config['last_type'] = 'dense'
#     [x, y] = dense_retrain_SL(config=model_config)
#     Results["lay"].extend(['Dense-Retrain'] * len(x))
#     Results["percentile"].extend(x)
#     Results["val_accuracy"].extend(y)

# for i in range(N):
#     print('Spectral Val/Vec...\n')
#     model_config['type'] = ['spec']
#     model_config['last_type'] = 'spec'
#     model_config['is_base'] = [False]
#     [x, y] = val_vec_train_trim(config=model_config)
#     Results["lay"].extend(['Alternate'] * len(x))
#     Results["percentile"].extend(x)
#     Results["val_accuracy"].extend(y)

for i in range(N):
    print('Dense...\n')
    model_config['type'] = ['dense']
    model_config['last_type'] = 'dense'
    dense_mod = train_model(config=model_config)
    [x, y] = dense_connectivity_trim_SL(dense_mod)
    Results["lay"].extend(['dense'] * len(x))
    Results["percentile"].extend(x)
    Results["val_accuracy"].extend(y)


accuracy_perc_plot = sb.lineplot(x="percentile", y="val_accuracy", hue="lay", style="lay",
                                 markers=True, dashes=False, ci="sd", data=Results)
accuracy_perc_plot.get_figure().savefig("./test/confr_reg_tanh.png")
plt.show()

# f = open("./fminst_reg_tanh.p", "wb")
# pk.dump(Results, f)
