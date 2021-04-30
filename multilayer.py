from Funzioni import *
import numpy as np
from ray import tune

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_config = {
    'type': 'Spectral',
    'activation': 'tanh',  # Activation function
    'hidden_size': 800,
    'hidden_layers': 2,

    # Training Parameters
    'dataset' : 'MNIST',
    'batch_size': 200,
    'epochs': 100,
    'save_path': 'C:\\Users\\loren\\PycharmProjects\\Git_Trimming\\MultiLayer_Results\\',
    'result_file_name': 'multilayer2_results-df.pk',
    'learn_rate': 0.001,

    # Trimming parameters
    'percentiles': np.arange(5, 105, 5)
}

activ_list = ['elu','relu','tanh']
type_list = ['Spectral', 'Dense']
dataset_list = ['MNIST', 'Fashion-MNIST']

model_config['type'] = tune.grid_search(type_list)
model_config['activation'] = tune.grid_search(activ_list)
model_config['dataset'] = tune.grid_search(dataset_list)

n_boot = 5

trialResources = {'cpu': 5, 'gpu': 0.4}
result = tune.run(train_and_trim_multilayer,
                  num_samples=n_boot,
                  resources_per_trial=trialResources,
                  config=model_config)
#%%
from Funzioni import *
dataset_list = ['MNIST', 'Fashion-MNIST']
activ_list = ['tanh', 'elu', 'relu']

plot_based_on(dataset='MNIST', activation='elu', fname='multilayer3_results-df.pk')