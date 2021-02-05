#!/bin/bash

python=$(which python)

echo -e "ALTERNATE"
echo -e "RELU"
$python -u cifar10_alternate.py --spectral_act relu -na 2
echo -e "ELU"
$python -u cifar10_alternate.py --spectral_act elu -na 2
echo -e "TANH"
$python -u cifar10_alternate.py --spectral_act tanh -na 2
echo -e "\n"

echo -e "SPECTRAL"
echo -e "RELU"
$python -u cifar10_spectral.py --spectral_act relu
echo -e "ELU"
$python -u cifar10_spectral.py --spectral_act elu
echo -e "TANH"
$python -u cifar10_spectral.py --spectral_act tanh
echo -e "\n"

echo -e "CONNECTIVITY"
echo -e "RELU"
$python -u cifar10_connectivity.py --activation relu
echo -e "ELU"
$python -u cifar10_connectivity.py --activation elu
echo -e "TANH"
$python -u cifar10_connectivity.py --activation tanh
echo -e "\n"
