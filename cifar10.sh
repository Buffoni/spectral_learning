#!/bin/bash

python=$(which python)

echo "SPECTRAL"
python -u cifar10_spectral.py
echo "CONNECTIVITY"
python -u cifar10_connectivity.py
echo "WEIGHT"
python -u cifar10_weight.py
