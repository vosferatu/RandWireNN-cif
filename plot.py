import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw_plot(epoch_list, train_loss_list, train_acc_list, val_acc_list):
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(epoch_list, train_loss_list, label='training loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(epoch_list, train_acc_list, label='train acc')
    plt.plot(epoch_list, val_acc_list, label='validation acc')
    plt.legend()

    if os.path.isdir('./plot'):
        plt.savefig('./plot/epoch_acc_plot.png')

    else:
        os.makedirs('./plot')
        plt.savefig('./plot/epoch_acc_plot.png')
    plt.close()


def plot_empirical():
    # c_78_p_0.0_graph_WS_dataset_CIFAR10_seed_1_name_empirical_one_opt_SGD_k_8.csv

    prob = [0.0, 0.15]

    for p in prob:
        ax = plt.gca()

        k_2 = pd.read_csv(
            f'./reporting/c_78_p_{p}_graph_WS_dataset_CIFAR10_seed_1_name_empirical_one_opt_SGD_k_2.csv')
        k_4 = pd.read_csv(
            f'./reporting/c_78_p_{p}_graph_WS_dataset_CIFAR10_seed_1_name_empirical_one_opt_SGD_k_4.csv')
        k_6 = pd.read_csv(
            f'./reporting/c_78_p_{p}_graph_WS_dataset_CIFAR10_seed_1_name_empirical_one_opt_SGD_k_6.csv')
        k_8 = pd.read_csv(
            f'./reporting/c_78_p_{p}_graph_WS_dataset_CIFAR10_seed_1_name_empirical_one_opt_SGD_k_8.csv')
        k_10 = pd.read_csv(
            f'./reporting/c_78_p_{p}_graph_WS_dataset_CIFAR10_seed_1_name_empirical_one_opt_SGD_k_10.csv')

        values = [k_2.accuracy.iloc[-1], k_4.accuracy.iloc[-1], k_6.accuracy.iloc[-1], k_8.accuracy.iloc[-1],
                  k_10.accuracy.iloc[-1]]

        ax.legend(['2', '4', '6', '8', '10'])

        x = np.arange(5)

        plt.bar(x, values)
        plt.xticks(x, ('k=2', 'k=4', 'k=6', 'k=8', 'k=10'))

        plt.title(f'CIFAR 10 - p={p}')

        plt.savefig(f'./plot/empirical_one_accuracy_CIFAR_10_{p}.png')
        plt.clf()

