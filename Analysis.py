import tensorflow as tf
from Functions import *

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_synchronous_execution(False)

spec_mod = train_model({'epochs': 30, 'type': ['spec']}, load_model=False)
spec_cp = spec_mod
[conn, acc_conn] = spectral_connectivity_trim(spec_cp)
[eig_lambda, acc_lambda] = spectral_eigval_trim(spec_mod)

dense_mod = train_model({'epochs': 30, 'type': ['dense']}, load_model=False)
[dense_conn, dense_acc_conn] = dense_connectivity_trim(dense_mod)

import matplotlib.pyplot as plt
plt.plot(eig_lambda, acc_lambda, 'bo', markersize=2)
plt.plot(conn, acc_conn, 'ro', markersize=2)
plt.plot(dense_conn, dense_acc_conn, 'go', markersize=2)
plt.show()


