import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_synchronous_execution(False)

from tqdm import tqdm
from SpectralLayer import Spectral
from tensorflow.keras.layers import Dense
import os
file_dir_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir_path)

import fnmatch
from pandas import DataFrame as df
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk


def Spectral_conf(size=2000,
                  is_base=True,
                  is_diag=True,
                  regularize=None,
                  is_bias=False,
                  activation=''):
    return {'units': size,
            'is_base_trainable': is_base,  # True means a trainable basis, False ow
            'is_diag_trainable': is_diag,  # True means a trainable eigenvalues, False ow
            'diag_regularizer': regularize,
            'use_bias': is_bias,  # True means a trainable bias, False ow
            'activation': activation
            }


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def Dense_conf(size=2000,
               use_bias=False,
               kernel_regularizer=None,
               activation=''):
    return {'units': size,
            'use_bias': use_bias,
            'kernel_regularizer': kernel_regularizer,
            'activation': activation
            }


def build_feedforward(model_config, multilayer=False, hidden_layers=2):
    """
    :param model_config: Guarda 'activation' e 'type'
    :return: modello compilato
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=784, dtype='float32'))
    while True:
        if model_config['type'] == 'Spectral':
            model.add(
                Spectral(**Spectral_conf(size=model_config['hidden_size'], activation=model_config['activation'])))
        elif model_config['type'] == 'Dense':
            model.add(Dense(**Dense_conf(size=model_config['hidden_size'], activation=model_config['activation'])))
        elif model_config['type'] == 'Alternate':
            model.add(Spectral(**Spectral_conf(size=model_config['hidden_size'], activation=model_config['activation'],
                                               is_base=False)))
        else:
            print("\nLayer type error\n")
            return -1
        hidden_layers -= 1

        if ((not multilayer) | hidden_layers == 0):
            break

    if model_config['type'] == 'Spectral':
        model.add(Spectral(**Spectral_conf(size=10, activation='softmax')))
    elif model_config['type'] == 'Dense':
        model.add(Dense(**Dense_conf(size=10, activation='softmax')))
    elif model_config['type'] == 'Alternate':
        model.add(Spectral(**Spectral_conf(size=10, activation='softmax', is_base=False)))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=model_config['learn_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  run_eagerly=False)
    return model


def load_dataset(name):
    if name == 'MNIST':
        datas = tf.keras.datasets.mnist
    elif name == 'Fashion-MNIST':
        datas = tf.keras.datasets.fashion_mnist
    else:
        print("\nDataset error\n")
        return -1

    (x_train, y_train), (x_test, y_test) = datas.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    flat_train = np.reshape(x_train, [x_train.shape[0], 28 * 28])
    flat_test = np.reshape(x_test, [x_test.shape[0], 28 * 28])
    return (flat_train, y_train), (flat_test, y_test)


def saving_file(model_config, test_results):
    path_name = model_config['save_path']
    os.makedirs(path_name, exist_ok=True)
    path_file = path_name + model_config['result_file_name']

    if not os.path.isfile(path_file):
        ris_df = df(columns=['dataset', 'activ', "type", "percentiles", "val_accuracy"])
        ris_df = ris_df.append(
            {'dataset': model_config['dataset'],
             'activ': model_config['activation'],
             "type": model_config['type'],
             "percentiles": test_results['percentiles'],
             "val_accuracy": test_results['val_accuracy']},
            ignore_index=True)
        with open(path_file, 'wb') as file:
            pk.dump(ris_df, file)
            print('\nScritto\n')

    else:
        with open(path_file, 'rb') as file:
            ris_df = pk.load(file)
            ris_df = ris_df.append(
                {'dataset': model_config['dataset'],
                 'activ': model_config['activation'],
                 "type": model_config['type'],
                 "percentiles": test_results['percentiles'],
                 "val_accuracy": test_results['val_accuracy']},
                ignore_index=True)
        with open(path_file, 'wb') as file:
            pk.dump(ris_df, file)


def spectral_trim(model, x_test, y_test, model_config):
    percentiles = model_config['percentiles']
    results = {'percentiles': [], "val_accuracy": []}
    diag = model.layers[0].diag.numpy()
    abs_diag = np.abs(diag)
    thresholds = [np.percentile(abs_diag, q=perc) for perc in percentiles]
    for t, perc in tqdm(list(zip(thresholds, percentiles)), desc="  Removing the eigenvalues"):
        diag[abs_diag < t] = 0.0
        model.layers[0].diag.assign(diag)
        test_results = model.evaluate(x_test, y_test, batch_size=1000, verbose=0)
        # storing the results
        results['percentiles'].append(perc)
        results["val_accuracy"].append(test_results[1])

    return results


def dense_trimming(model, x_test, y_test, model_config):
    percentiles = model_config['percentiles']
    weights = model.layers[0].weights[0].numpy()
    connectivity = np.abs(weights).sum(axis=0)
    thresholds = [np.percentile(connectivity, q=perc) for perc in percentiles]
    results = {'percentiles': [], 'val_accuracy': []}

    for t, perc in tqdm(list(zip(thresholds, percentiles)), desc="  Removing the nodes"):
        weights[:, connectivity < t] = 0.0
        model.layers[0].weights[0].assign(weights)
        test_results = model.evaluate(x_test, y_test, batch_size=1000, verbose=0)
        # storing the results
        results['percentiles'].append(perc)
        results["val_accuracy"].append(test_results[1])

    return results


def alternate_trimming(model, model_config, flat_train, y_train, flat_test, y_test):
    percentiles = model_config['percentiles']
    results = {"percentiles": [], "val_accuracy": []}

    diag = model.layers[0].diag.numpy()
    abs_diag = np.abs(diag)
    thresholds = [np.percentile(abs_diag, q=perc) for perc in percentiles]

    for t, perc in tqdm(list(zip(thresholds, percentiles)), desc="  Removing the eigenvalues and train vectors"):
        diag[abs_diag < t] = 0.0
        hid_size = np.count_nonzero(diag)

        # Smaller Model
        new_model = tf.keras.Sequential()
        new_model.add(tf.keras.layers.Input(shape=784, dtype='float32'))
        new_model.add(Spectral(**Spectral_conf(size=hid_size, activation=model_config['activation'], is_base=True)))
        new_model.add(Spectral(**Spectral_conf(size=10, activation='softmax', is_base=True)))

        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=model_config['learn_rate']),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'],
                          run_eagerly=False)

        new_model.layers[0].diag.assign(diag[diag != 0.0])
        tmp_base = model.layers[0].base.numpy()
        new_model.layers[0].base.assign(tmp_base[:, diag != 0.0])

        new_model.layers[1].diag.assign(model.layers[1].diag)
        tmp_base = model.layers[1].base.numpy()
        new_model.layers[1].base.assign(tmp_base[diag != 0.0, :])

        new_model.fit(flat_train, y_train, verbose=0, batch_size=model_config['batch_size'], epochs=20)
        test_results = new_model.evaluate(flat_test, y_test, batch_size=1000, verbose=0)
        results['percentiles'].append(perc)
        results["val_accuracy"].append(test_results[1])
        new_model = []

    return results


def train_and_trim(model_config):
    (flat_train, y_train), (flat_test, y_test) = load_dataset(model_config['dataset'])
    model = build_feedforward(model_config)
    model.fit(flat_train, y_train, verbose=0, batch_size=model_config['batch_size'], epochs=model_config['epochs'])
    if model_config['type'] == 'Spectral':
        saving_file(model_config, spectral_trim(model, flat_test, y_test, model_config))
    if model_config['type'] == 'Dense':
        saving_file(model_config, dense_trimming(model, flat_test, y_test, model_config))
    if model_config['type'] == 'Alternate':
        saving_file(model_config, alternate_trimming(model, model_config, flat_train, y_train, flat_test, y_test))


from os.path import join


def train_and_trim_multilayer(model_config):
    (flat_train, y_train), (flat_test, y_test) = load_dataset(model_config['dataset'])

    model = build_feedforward(model_config, multilayer=True, hidden_layers=model_config['hidden_layers'])
    model.fit(flat_train, y_train, verbose=0, batch_size=model_config['batch_size'], epochs=model_config['epochs'])

    if model_config['type'] == 'Spectral':
        saving_file(model_config, spectral_trim_ML(model, flat_test, y_test, model_config))
    if model_config['type'] == 'Dense':
        saving_file(model_config, dense_trimming_ML(model, flat_test, y_test, model_config))
    if model_config['type'] == 'Alternate':
        saving_file(model_config, alternate_trimming_ML(model, model_config, flat_train, y_train, flat_test, y_test))


def plot_based_on(dataset='MNIST', activation='tanh', fname='result_dataframe.pk', lable_size=13, ticks_size=11
                  ,multilayer=False,save_fig=True):
    path = find(fname, os.getcwd())

    with open(path[0], 'rb') as f:
        df = pk.load(f)

    dataset_mask = df['dataset'] == dataset
    activation_mask = df['activ'] == activation

    to_plot = df[dataset_mask & activation_mask]

    plt.figure(figsize=(5.5, 5))
    sb.lineplot(x="percentiles", y="val_accuracy", hue='type',
                palette={'Alternate': 'green', 'Spectral': 'blue', 'Dense': 'orange'}, style="type",
                markers=True, dashes=False, ci="sd", data=plot_preprocess(to_plot))
    plt.title(dataset + ' - Activation:' + activation)
    lbl = {'fontsize': lable_size}
    tsz = {'fontsize': ticks_size}
    plt.xlabel('Percentile', **lbl)
    plt.xticks(**tsz)
    plt.yticks(**tsz)
    plt.ylabel('Val. Accuracy', **lbl)
    plt.legend(**lbl)

    if save_fig:
        if multilayer:
            folder = join(os.getcwd(), 'Figures trimming multi', dataset)
        else:
            folder = join(os.getcwd(), 'Figures trimming', dataset)
        os.makedirs(folder, exist_ok=True)
        save_path = join(folder, activation)
        plt.savefig(save_path)
    plt.show()


def plot_preprocess(dati):
    dati = dati[['type', 'percentiles', 'val_accuracy']].reset_index(drop=True)
    ris = {"type": [], "percentiles": [], "val_accuracy": []}

    for i in range(len(dati)):
        ris["type"].extend([dati['type'][i]] * len(dati['val_accuracy'][i]))
        ris["percentiles"].extend(dati['percentiles'][i])
        ris["val_accuracy"].extend(dati['val_accuracy'][i])

    return ris


def spectral_trim_ML(model, flat_train, y_test, model_config):
    percentiles = model_config['percentiles']
    results = {'percentiles': [], "val_accuracy": []}

    for i in range(len(model.layers) - 1):
        if i == 0:
            autov_list = model.layers[i].diag.numpy()
        else:
            autov_list = np.append(autov_list, model.layers[i].diag.numpy(), 0)

    diag = [model.layers[i].diag.numpy() for i in range(len(model.layers) - 1)]

    abs_diag = np.abs(autov_list)
    thresholds = [np.percentile(abs_diag, q=perc) for perc in percentiles]
    for t, perc in tqdm(list(zip(thresholds, percentiles)), desc="  Removing the eigenvalues"):
        for i in range(len(model.layers) - 1):
            diag[i][abs(diag[i]) < t] = 0.0
            model.layers[i].diag.assign(diag[i])

        test_results = model.evaluate(flat_train, y_test, batch_size=1000, verbose=0)
        # storing the results
        results['percentiles'].append(perc)
        results["val_accuracy"].append(test_results[1])

    return results


def dense_trimming_ML(model, x_test, y_test, model_config):
    percentiles = model_config['percentiles']
    results = {'percentiles': [], 'val_accuracy': []}

    weights = [model.layers[i].weights[0].numpy() for i in range(len(model.layers) - 1)]
    connectivity = [np.abs(weights[i]).sum(axis=0) for i in range(len(model.layers) - 1)]

    for i in range(len(model.layers) - 1):
        if i == 0:
            conn_list = np.abs(weights[i]).sum(axis=0)
        else:
            conn_list = np.append(conn_list, np.abs(weights[i]).sum(axis=0), 0)

    thresholds = [np.percentile(conn_list, q=perc) for perc in percentiles]

    for t, perc in tqdm(list(zip(thresholds, percentiles)), desc="  Removing the nodes"):
        for i in range(len(model.layers) - 1):
            weights[i][:, connectivity[i] < t] = 0.0
            model.layers[i].weights[0].assign(weights[i])

        test_results = model.evaluate(x_test, y_test, batch_size=1000, verbose=0)

        # storing the results
        results['percentiles'].append(perc)
        results["val_accuracy"].append(test_results[1])

    return results


def alternate_trimming_ML(model, model_config, flat_train, y_train, flat_test, y_test):
    percentiles = model_config['percentiles']
    results = {'percentiles': [], "val_accuracy": []}

    for i in range(len(model.layers) - 1):
        if i == 0:
            autov_list = model.layers[i].diag.numpy()
        else:
            autov_list = np.append(autov_list, model.layers[i].diag.numpy(), 0)

    diag = [model.layers[i].diag.numpy() for i in range(len(model.layers) - 1)]

    abs_diag = np.abs(autov_list)
    thresholds = [np.percentile(abs_diag, q=perc) for perc in percentiles]

    for t, perc in tqdm(list(zip(thresholds, percentiles)), desc="  Removing the eigenvalues and train vectors"):
        hid_size = np.zeros(len(model.layers) - 1)
        for i in range(len(model.layers) - 1):
            diag[i][abs(diag[i]) < t] = 0.0
            hid_size[i] = np.count_nonzero(diag[i])

        # Smaller Model
        new_model = tf.keras.Sequential()
        new_model.add(tf.keras.layers.Input(shape=784, dtype='float32'))

        for i in range(len(model.layers) - 1):
            new_model.add(
                Spectral(**Spectral_conf(size=hid_size[i], activation=model_config['activation'], is_base=True)))

        new_model.add(Spectral(**Spectral_conf(size=10, activation='softmax', is_base=True)))

        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=model_config['learn_rate']),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'],
                          run_eagerly=False)

        for i in range(len(model.layers) - 1):
            new_model.layers[i].diag.assign(diag[i][diag[i] != 0.0])
            tmp_base = model.layers[i].base.numpy()
            if i == 0:
                new_model.layers[i].base.assign(tmp_base[:, diag[i] != 0.0])
            else:
                ind1 = diag[i - 1] != 0.0
                ind2 = diag[i] != 0.0
                tmp_base = tmp_base[ind1, :]
                tmp_base = tmp_base[:, ind2]
                new_model.layers[i].base.assign(tmp_base)
        i += 1
        new_model.layers[i].diag.assign(model.layers[i].diag)
        tmp_base = model.layers[i].base.numpy()
        new_model.layers[i].base.assign(tmp_base[diag[i - 1] != 0.0, :])

        new_model.fit(flat_train, y_train, verbose=0, batch_size=model_config['batch_size'], epochs=20)
        test_results = new_model.evaluate(flat_test, y_test, batch_size=1000, verbose=0)
        results['percentiles'].append(perc)
        results["val_accuracy"].append(test_results[1])

    return results
