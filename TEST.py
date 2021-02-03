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
    'batch_size': 500,
    'epochs': 30,

}

plt.figure(0, dpi=200)

Results = {"lay": [], "percentile": [], "val_accuracy": []}

print('Spectral...\n')
model_config['type'] = ['spec']
model_config['is_base'] = [True]
model_config['last_type'] = 'spec'
model = train_model(config=model_config)
f = open('testset.pickle', 'rb')
x_test, y_test = pk.load(f)
f.close()

base = model.layers[0].base.numpy()
diag = model.layers[0].diag.numpy()
weights = -mult(base, diag)
connectivity = np.sum(weights, axis=0)
percentiles = list(perc_span)
results = {"percentile": [], "val_accuracy": []}
thresholds = [np.percentile(abs(diag[connectivity < 0]), q=perc) for perc in percentiles]
1
model.evaluate(x_test, y_test, batch_size=1000, verbose=1)




#%%
diag[connectivity < 0] = 0.0
k = np.count_nonzero(connectivity < 0)
model.layers[0].diag.assign(diag)
test_results = model.evaluate(x_test, y_test, batch_size=1000, verbose=0)
results["percentile"].append(100 * k / T)
results["val_accuracy"].append(test_results[1])
#%%
for t, perc in tqdm(list(zip(thresholds, percentiles)[1:]), desc="  Removing the nodes"):
    if ind == 1:
        diag[connectivity < 0] = 0.0
        k = np.count_nonzero(connectivity < 0)
        model.layers[0].diag.assign(diag)
        test_results = model.evaluate(x_test, y_test, batch_size=1000, verbose=0)
        results["percentile"].append(100 * k / T)
        results["val_accuracy"].append(test_results[1])
        ind = 0
    else:
        diag[abs(diag) < t] = 0.0
        model.layers[0].diag.assign(diag)
        test_results = model.evaluate(x_test, y_test, batch_size=1000, verbose=0)
        results["percentile"].append(100 * k / T + perc * (T - k) / T)
        results["val_accuracy"].append(test_results[1])

#%%
diag[connectivity < 0.0] = 0.0
model.layers[0].diag.assign(diag)

model.evaluate(x_test, y_test, batch_size=1000, verbose=1)

