# Spectral Layer
This repository contains the code to reproduce some of the results in: https://www.nature.com/articles/s41467-021-21481-0
The layer implements a linear transfer between layers representing it on the eigenvectors and eigenvalues of the network constituted by layer k and k+1, where k runs in the number of layers.
The module ```SpectralLayer.py``` contains an implementation of the spectral layer class that can be easily embedded in every Tensorflow model.

The notebook ```Spactral_Layer_example.ipynb``` contains a usage example.

Dependecies:
```
tensorflow > 2.0
numpy
matplotlib
```

