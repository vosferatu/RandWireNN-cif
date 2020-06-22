#!/usr/bin/env bash

#python dataset_potential.py  --dataset-mode CIFAR100 --name frozen_2

python dataset_potential.py  --opt 1 --dataset-mode CIFAR100 --is-train False --load-model True --name frozen_2 --freeze True

python dataset_potential.py  --optimizer SGD_NO_MOMENTUM --opt 1 --dataset-mode CIFAR100 --is-train False --load-model True --name frozen_2 --freeze True