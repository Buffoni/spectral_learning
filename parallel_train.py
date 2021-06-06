from ray import tune
from Funzioni import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_config = {
    'type': 'Spectral',
    'activation': 'tanh',  # Activation function
    'hidden_layers': 1,
    'hidden_size': 2000,

    # Training Parameters
    'dataset' : 'MNIST',
    'batch_size': 200,
    'epochs': 100,
    'save_path': 'C:\\Users\\loren\\PycharmProjects\\Git_Trimming\\Results\\',
    'result_file_name': 'results-df.pk',
    'learn_rate': 0.001,

    # Trimming parameters
    'percentiles': np.arange(5, 105, 5)
}

activ_list = ['elu']
type_list = ['Spectral', 'Dense']
dataset_list = ['Fashion-MNIST']

model_config['type'] = tune.grid_search(type_list)
model_config['activation'] = tune.grid_search(activ_list)
model_config['dataset'] = tune.grid_search(dataset_list)

n_boot = 5

trialResources = {'cpu': 5, 'gpu': 0.4}
result = tune.run(train_and_trim,
                  num_samples=n_boot,
                  resources_per_trial=trialResources,
                  config=model_config)

#%%
from Funzioni import *
dataset_list = ['MNIST', 'Fashion-MNIST']
activ_list = ['tanh', 'elu', 'relu']

plot_based_on(dataset='Fashion-MNIST', activation='tanh', fname='.pk')
#%%
for ds in dataset_list:
    for act in activ_list:
        plot_based_on(dataset=ds, activation=act,fname='results_df.pk')
#%%
from Funzioni import *
import pickle as pk
path = find('results_df.pk', os.getcwd())

with open(path[0], 'rb') as f:
    df = pk.load(f)
#%%
dataset_mask = df['dataset'] == 'MNIST'
activation_mask = df['activ'] == 'relu'
type_mask =  df['type'] != 'Alternate'
mask = dataset_mask & activation_mask & type_mask
new = df[~mask]
#%%
with open(os.getcwd()+'\\Results\\results_df.pk', 'wb') as f:
    df = pk.dump(new,f)

