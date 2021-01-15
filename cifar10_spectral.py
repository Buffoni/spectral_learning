import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time, sleep
from SpectralLayer import Spectral
from skimage.transform import resize
from tensorflow.linalg import matmul as tmm
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

plt.rcParams["figure.figsize"] = (20, 10)

# reproducibility
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# defining the training, validation, test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,
#                                                   stratify=y_train, random_state=random_seed)
x_train = np.asarray([preprocess_input(resize(img, output_shape=[96, 96])) for img in tqdm(x_train)])
# x_val = np.asarray([preprocess_input(resize(img, output_shape=[96, 96])) for img in tqdm(x_val)])
x_test = np.asarray([preprocess_input(resize(img, output_shape=[96, 96])) for img in tqdm(x_test)])

def create_net(nclasses=10, spectral_out=512, diag_reg=0.01, learning_rate=0.001):
  backbone = tf.keras.applications.MobileNetV2(input_shape=[96, 96, 3], include_top=False, weights='imagenet')
  backbone.trainable = False
  net = tf.keras.Sequential()
  net.add(backbone)
  net.add(tf.keras.layers.GlobalMaxPool2D())
  net.add(Spectral(spectral_out, 
                   use_bias=False,
                   activation="relu", 
                   diag_regularizer=tf.keras.regularizers.l1(l1=diag_reg)))
  net.add(tf.keras.layers.Dense(nclasses, activation="softmax"))
  net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"],
              run_eagerly=False)
  return net

nattempts = 3
percentiles = list(range(0, 110, 10))
diag_reg = [0, 1e-5, 5e-4, 1e-4, 1e-3, 1e-2, 1e-1, 1]
hyperparameters = {"epochs": [25, 50],
                   "lr": [1e-4, 1e-3]}

results = {"diag_reg": [], "percentile": [], "val_accuracy": []}
for counter, dr in enumerate(diag_reg):
  print("{}-th training (of {}) with diagreg = {}".format(counter+1, len(diag_reg), dr))
  tic = time()
  model = KerasClassifier(build_fn=lambda lr: create_net(learning_rate=lr, diag_reg=dr),
                          epochs=10, batch_size=128, verbose=0)
  model = GridSearchCV(estimator=model, param_grid=hyperparameters, n_jobs=1, cv=3, refit=False)
  best_params = model.fit(x_train, y_train).best_params_
  print("Grid Search done in {:.3f} secs".format(time()-tic))
  for attempt in range(nattempts):
    tic = time()
    model = create_net(learning_rate=best_params["lr"], diag_reg=dr)
    model.fit(x_train, y_train, epochs=best_params["epochs"], batch_size=128, verbose=0)
    print("  {}-th training (of {}) done in {:.3f} secs".format(attempt+1, nattempts, time()-tic))
    diag = model.layers[2].diag.numpy()
    abs_diag = np.abs(diag)
    thresholds = [np.percentile(abs_diag, q=perc) for perc in percentiles]
    for t, perc in tqdm(list(zip(thresholds, percentiles)), desc="  Removing the eigenvalues"):
      diag[abs_diag < t] = 0.0
      model.layers[2].diag.assign(diag)
      test_results = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
      # storing the results
      results["diag_reg"].append("{}".format(dr))
      results["percentile"].append(perc)
      results["val_accuracy"].append(test_results[1])

results = pd.DataFrame(results)
results.to_csv("./test/s_accuracy_perc_cifar10.csv", index=False)
print(results)
accuracy_perc_plot = sns.lineplot(x="percentile", y="val_accuracy", hue="diag_reg", style="diag_reg", 
                                  markers=True, dashes=False, ci="sd", data=results)
accuracy_perc_plot.get_figure().savefig("./test/s_accuracy_perc_cifar10.png")
