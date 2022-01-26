# Spectral Pruning
This repository contains the code to reproduce some of the results in: https://arxiv.org/pdf/2108.00940.pdf.
Different spectral pruning techniques are implemented acting on the direct space of the connections (as a benchmark) and in the reciprocal space.
While acting in the reciprocal space we exploit the already implemented spectral layer in the following way:
the eigenvalues are trained and then cosidered as a proxy for the importance of every node in the network. By doing so we can prune a desired amount of nodes basing on such indicator.
In the direct space, as a comparison, the importance of every node is based on the absolute incoming connectivity.
In <em>multilayer_pruning.ipynb</em> the tests for MNIST and Fashion-MNIST dataset are implemented and shown whereas in ,em>cifar10*.py</em> files the same procedures are applied to the CIFAR10 dataset. For the altter case a feature extraction using MobileNetV2 is done and the pruning is done only in the last part of the network.

Dependecies:
```
tensorflow > 2.3
numpy
matplotlib
```

### Coming Soon
Implementation of the pruning algorithm as a tool to apply to a fully trained (with spectral layers) Neural Network.

