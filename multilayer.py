import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from Funzioni import *
import numpy as np
from ray import tune

hidden_layers = 3

model_config = {
    'type': 'Spectral',
    'activation': 'tanh',  # Activation function
    'hidden_size': 500,
    'hidden_layers': hidden_layers,

    # Training Parameters
    'dataset': 'Fashion-MNIST',
    'batch_size': 200,
    'epochs': 100,
    'save_path': '/home/lorenzogiambagli/Documents/Trimming/MultiLayer_Results/', #'C:\\Users\\loren\\PycharmProjects\\Git_Trimming\\MultiLayer_Results\\',
    'result_file_name': 'multilayer' + str(hidden_layers) + '_results-df.pk',
    'learn_rate': 0.001,

    # Trimming parameters
    'percentiles': np.arange(5, 100, 5)
}

activ_list = ['elu']#['relu', 'tanh']
type_list = ['Alternate']
dataset_list = [ 'Fashion-MNIST']

model_config['type'] = tune.grid_search(type_list)
model_config['activation'] = tune.grid_search(activ_list)
model_config['dataset'] = tune.grid_search(dataset_list)

n_boot = 5
trialResources = {'cpu': 1, 'gpu': 0.2}

# test = train_and_trim_multilayer(model_config)
result = tune.run(train_and_trim_multilayer,
                  num_samples=n_boot,
                  resources_per_trial=trialResources,
                  config=model_config)

#%%
# from Funzioni import *

# dataset_list = ['MNIST', 'Fashion-MNIST']
# activ_list = ['tanh', 'elu', 'relu']
# for ds in dataset_list:
#     for act in activ_list:
#         plot_based_on(dataset=ds,save_fig=True, activation=act, fname='multilayer3_results-df.pk')
#%%

# from Funzioni import *
# model_config = {
#     'type': 'Spectral',
#     'activation': 'tanh',  # Activation function
#     'hidden_size': 300,
#     'hidden_layers': 2,
#
#     # Training Parameters
#     'dataset': 'Fashion-MNIST',
#     'batch_size': 200,
#     'epochs': 2,#120,
#     'save_path': 'C:\\Users\\loren\\PycharmProjects\\Git_Trimming\\MultiLayer_Results\\',
#     'result_file_name': 'lay' + str(3) + '_results-df.pk',
#     'learn_rate': 0.001,
#
#     # Trimming parameters
#     'percentiles': np.arange(90, 105, 5)
# }
# (flat_train, y_train), (flat_test, y_test) = load_dataset(model_config['dataset'])
# model = build_feedforward(model_config, multilayer=True, hidden_layers=model_config['hidden_layers'])
# model.fit(flat_train, y_train, verbose=1, batch_size=model_config['batch_size'], epochs=model_config['epochs'])
#
# percentiles = model_config['percentiles']
# results = {'percentiles': [], "val_accuracy": []}
#
# for i in range(len(model.layers) - 1):
#     if i == 0:
#         autov_list = model.layers[i].diag.numpy()
#     else:
#         autov_list = np.append(autov_list, model.layers[i].diag.numpy(), 0)
#
# diag = [model.layers[i].diag.numpy() for i in range(len(model.layers) - 1)]
#
# abs_diag = np.abs(autov_list)
# thresholds = [np.percentile(abs_diag, q=perc) for perc in percentiles]
# for t, perc in tqdm(list(zip(thresholds, percentiles)), desc="  Removing the eigenvalues and train vectors"):
#     hid_size = np.zeros(len(model.layers) - 1)
#     for i in range(len(model.layers) - 1):
#         diag[i][abs(diag[i]) < t] = 0.0
#         hid_size[i] = np.count_nonzero(diag[i])
#
#     # Smaller Model
#     new_model = tf.keras.Sequential()
#     new_model.add(tf.keras.layers.Input(shape=784, dtype='float32'))
#
#     for i in range(len(model.layers) - 1):
#         new_model.add(Spectral(**Spectral_conf(size=hid_size[i], activation=model_config['activation'], is_base=True)))
#
#     new_model.add(Spectral(**Spectral_conf(size=10, activation='softmax', is_base=True)))
#
#     new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=model_config['learn_rate']),
#                       loss='sparse_categorical_crossentropy',
#                       metrics=['accuracy'],
#                       run_eagerly=False)
#
#     for i in range(len(model.layers) - 1):
#         new_model.layers[i].diag.assign(diag[i][diag[i] != 0.0])
#         tmp_base = model.layers[i].base.numpy()
#         if i == 0:
#             new_model.layers[i].base.assign(tmp_base[:, diag[i] != 0.0])
#         else:
#             ind1 = diag[i - 1] != 0.0
#             ind2 = diag[i] != 0.0
#             tmp_base = tmp_base[ind1, :]
#             tmp_base = tmp_base[:, ind2]
#             new_model.layers[i].base.assign(tmp_base)
#     i+=1
#     new_model.layers[i].diag.assign(model.layers[i].diag)
#     tmp_base = model.layers[i].base.numpy()
#     new_model.layers[i].base.assign(tmp_base[diag[i-1] != 0.0, :])
#
#     new_model.fit(flat_train, y_train, verbose=0, batch_size=model_config['batch_size'], epochs=20)
#     test_results = new_model.evaluate(flat_test, y_test, batch_size=1000, verbose=0)
#     results['percentiles'].append(perc)
#     results["val_accuracy"].append(test_results[1])