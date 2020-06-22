#!/usr/bin/env bash
python dataset_potential.py  --dataset-mode MNIST --name frozen_2

python dataset_potential.py  --opt 1 --dataset-mode MNIST --is-train False --load-model True --name frozen_2 --freeze True

python dataset_potential.py  --optimizer SGD_NO_MOMENTUM --opt 1 --dataset-mode MNIST --is-train False --load-model True --name frozen_2 --freeze True



python dataset_potential.py  --dataset-mode FASHION_MNIST --name frozen_2

python dataset_potential.py  --opt 1 --dataset-mode FASHION_MNIST --is-train False --load-model True --name frozen_2 --freeze True

python dataset_potential.py  --optimizer SGD_NO_MOMENTUM --opt 1 --dataset-mode FASHION_MNIST --is-train False --load-model True --name frozen_2 --freeze True



python dataset_potential.py  --dataset-mode CIFAR10 --name frozen_2

python dataset_potential.py  --opt 1 --dataset-mode CIFAR10 --is-train False --load-model True --name frozen_2 --freeze True

python dataset_potential.py  --optimizer SGD_NO_MOMENTUM --opt 1 --dataset-mode CIFAR10 --is-train False --load-model True --name frozen_2 --freeze True


