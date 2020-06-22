#!/usr/bin/env bash

sudo python dataset_potential.py  --dataset-mode IMAGENET --name no_freeze --epochs 0

sudo python dataset_potential.py  --opt 1 --dataset-mode IMAGENET --is-train False --load-model True --name no_freeze --epochs 0

sudo python dataset_potential.py  --epochs 0 --optimizer SGD_NO_MOMENTUM --opt 1 --dataset-mode IMAGENET --is-train False --load-model True --name no_freeze

