
import tensorflow as tf
from Functions import *
from tensorflow.keras import Model
from tensorflow.keras import layers
from SpectralLayer import Spectral
import matplotlib.pyplot as plt
import seaborn as sb

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Parallel execution's stuff
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_synchronous_execution(False)

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='elu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(784, activation='softmax'),
      layers.Reshape((28, 28))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

class SpecEncoder(Model):
  def __init__(self, latent_dim):
    super(SpecEncoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      Spectral(latent_dim, activation='elu', is_base_trainable=True, is_diag_trainable=True),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(784, activation='softmax'),
      layers.Reshape((28, 28))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

hid_dim = 750

plt.figure(0, dpi=200)

Results = {"lay": [], "percentile": [], "val_loss": []}

N = 5

for i in range(N):
    autoencoder = SpecEncoder(hid_dim)
    autoencoder.compile(optimizer='adam', loss='MSE')
    autoencoder.fit(x_train, x_train,
                    epochs=30,
                    batch_size=100,
                    verbose=0)
    print('Spectral...\n')
    [x, y] = spectral_encoder_SL(autoencoder)
    Results["lay"].extend(['Spectral'] * len(x))
    Results["percentile"].extend(x)
    Results["val_loss"].extend(y)

for i in range(N):
    autoencoder = Autoencoder(hid_dim)
    autoencoder.compile(optimizer='adam', loss='MSE')
    autoencoder.fit(x_train, x_train,
                    epochs=30,
                    batch_size=100,
                    verbose=0)

    print('Dense...\n')
    [x, y] = dense_encoder_trim_SL(autoencoder)
    Results["lay"].extend(['Dense'] * len(x))
    Results["percentile"].extend(x)
    Results["val_loss"].extend(y)

accuracy_perc_plot = sb.lineplot(x="percentile", y="val_loss", hue="lay", style="lay",
                                 markers=True, dashes=False, ci="sd", data=Results)
accuracy_perc_plot.get_figure().savefig("./test/mnist_enc_elu.png")
plt.show()

#%%
import matplotlib.pyplot as plt
encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()