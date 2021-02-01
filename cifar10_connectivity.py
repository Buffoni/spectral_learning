import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from SpectralLayer import Spectral
from skimage.transform import resize
from tensorflow.linalg import matmul as tmm
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# tensorflow
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# matplotlib
plt.rcParams["figure.figsize"] = (20, 10)
# reproducibility
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
# environment
os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-civitelli"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def get_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,
    #                                                   stratify=y_train, random_state=random_seed)
    x_train = np.asarray([preprocess_input(resize(img, output_shape=[96, 96])) for img in tqdm(x_train, desc="Train")])
    # x_val = np.asarray([preprocess_input(resize(img, output_shape=[96, 96])) for img in tqdm(x_val, desc="Val")])
    x_test = np.asarray([preprocess_input(resize(img, output_shape=[96, 96])) for img in tqdm(x_test, desc="Test")])

    # features extraction w/ MobileNetV2
    features_extractor = tf.keras.Sequential([tf.keras.applications.MobileNetV2(input_shape=[96, 96, 3], include_top=False,
                                                                                weights="imagenet"),
                                              tf.keras.layers.GlobalMaxPool2D()])
    x_train = features_extractor.predict(x=x_train, verbose=1)
    # x_val = features_extractor.predict(x=x_val, verbose=1)
    x_test = features_extractor.predict(x=x_test, verbose=1)
    return x_train, y_train, x_test, y_test

def create_net(in_dim, activation, nclasses=10, out_dim=512, regularizer=0.01, learning_rate=0.001):
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Input(shape=(in_dim), dtype="float32"))
    net.add(tf.keras.layers.Dense(out_dim,
                                  use_bias=False,
                                  activation=activation,
                                  kernel_regularizer=tf.keras.regularizers.l1(l1=regularizer)))
    net.add(tf.keras.layers.Dense(nclasses, activation="softmax"))
    net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    return net

def main(n_attempts=5, activation="relu", batch_size=32):
    percentiles = list(range(0, 110, 10))
    regularizers = [0, 1e-5, 5e-4, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    hyperparameters = {"epochs": [20, 40, 60], "lr": [1e-4, 1e-3]}

    x_train, y_train, x_test, y_test = get_data()

    results = {"regularizer": [], "percentile": [], "test_accuracy": []}
    for counter, reg in enumerate(regularizers):
        print("{}-th training (of {}) with regularizer = {}".format(counter+1, len(regularizers), reg))
        tic = time()
        model = KerasClassifier(build_fn=lambda lr: create_net(learning_rate=lr,
                                                               regularizer=reg,
                                                               in_dim=x_train.shape[1],
                                                               activation=activation),
                                epochs=25, batch_size=batch_size, verbose=0)
        model = GridSearchCV(estimator=model, param_grid=hyperparameters, n_jobs=1, cv=3, verbose=2, refit=False)
        best_params = model.fit(x_train, y_train).best_params_
        print("Grid Search done in {:.3f} secs".format(time()-tic))
        for attempt in range(n_attempts):
            print(f"  {attempt+1}-th training (of {n_attempts})")
            model = create_net(learning_rate=best_params["lr"], regularizer=reg, 
                               in_dim=x_train.shape[1], activation=activation)
            model.fit(x_train, y_train, epochs=best_params["epochs"], batch_size=batch_size, verbose=1)
            kernel = model.layers[0].weights[0].numpy()
            connectivity = np.abs(kernel).sum(axis=1)
            thresholds = [np.percentile(connectivity, q=perc) for perc in percentiles]
            for t, perc in tqdm(zip(thresholds, percentiles), total=len(thresholds), desc="  Removing the eigenvalues"):
                kernel[connectivity < t, :] = 0.0
                model.layers[0].weights[0].assign(kernel)
                test_results = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
                # storing the results
                results["regularizer"].append(f"{reg}")
                results["percentile"].append(perc)
                results["test_accuracy"].append(test_results[1])

    results = pd.DataFrame(results)
    print(results)
    
    opath = os.path.join("./test", f"connectivity_abs_{activation}")
    results.to_csv(f"{opath}.csv", index=False)
    sns.lineplot(x="percentile", y="test_accuracy",
                 hue="regularizer", style="regularizer",
                 markers=True, dashes=False, ci="sd", 
                 data=results).get_figure().savefig(f"{opath}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-act", "--activation", type=str, default="relu")
    parser.add_argument("-na", "--number_of_attempts", type=int, default=5)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    main(n_attempts=args.number_of_attempts, activation=args.activation, batch_size=args.batch_size)
