import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

datasets = ['CIFAR10', 'MNIST', 'FASHION_MNIST', 'CIFAR100']


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


def plot_node_study_time():
    # c_78_p_0.0_graph_WS_dataset_CIFAR10_seed_1_name_empirical_one_opt_SGD_k_8.csv
    nodes_check = [6, 9, 12, 16, 20, 24, 28, 32]

    for dataset in datasets:
        values_ER = []
        for num in nodes_check:
            file_ER = pd.read_csv(
                f'./reporting/node_study/c_78_p_0.2_graph_ER_dataset_{dataset}_seed_1_name_empirical_nodes_opt_SGD_nodes_{num}.csv')
            values_ER.append(round(file_ER.time.iloc[-1], 1))

        values_WS = []
        for num in nodes_check:
            file_WS = pd.read_csv(
                f'./reporting/node_study/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_1_name_empirical_nodes_opt_SGD_nodes_{num}.csv')
            values_WS.append(round(file_WS.time.iloc[-1], 1))

        values_BA = []
        for num in nodes_check:
            file_BA = pd.read_csv(
                f'./reporting/node_study/c_78_p_0.75_graph_BA_dataset_{dataset}_seed_1_name_empirical_nodes_opt_SGD_nodes_{num}.csv')
            values_BA.append(round(file_BA.time.iloc[-1], 1))

        node_num = ['6', '9', '12', '16', '20', '24', '28', '32']

        fig = go.Figure(data=[
            go.Bar(name='WS', x=node_num, y=values_WS, text=values_WS, textposition='auto'),
            go.Bar(name='ER', x=node_num, y=values_ER, text=values_ER, textposition='auto'),
            go.Bar(name='BA', x=node_num, y=values_BA, text=values_BA, textposition='auto')
        ])
        # Change the bar mode
        fig.update_layout(
            title=f'{dataset} node number time performance',
            title_x=0.5,
            xaxis_title="Number of nodes",
            yaxis_title="Time [s]",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#000000"
            ),
            barmode='group'
        )
        fig.write_image(f"plot/{dataset}_node_time_evaluation.pdf")
        fig.show()


def plot_node_study_accuracy():
    # c_78_p_0.0_graph_WS_dataset_CIFAR10_seed_1_name_empirical_one_opt_SGD_k_8.csv
    nodes_check = [6, 9, 12, 16, 20, 24, 28, 32]

    for dataset in datasets:
        values_ER = []
        for num in nodes_check:
            file_ER = pd.read_csv(
                f'./reporting/node_study/c_78_p_0.2_graph_ER_dataset_{dataset}_seed_1_name_empirical_nodes_opt_SGD_nodes_{num}.csv')
            values_ER.append(round(file_ER.accuracy.iloc[-1], 2))

        values_WS = []
        for num in nodes_check:
            file_WS = pd.read_csv(
                f'./reporting/node_study/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_1_name_empirical_nodes_opt_SGD_nodes_{num}.csv')
            values_WS.append(round(file_WS.accuracy.iloc[-1], 2))

        values_BA = []
        for num in nodes_check:
            file_BA = pd.read_csv(
                f'./reporting/node_study/c_78_p_0.75_graph_BA_dataset_{dataset}_seed_1_name_empirical_nodes_opt_SGD_nodes_{num}.csv')
            values_BA.append(round(file_BA.accuracy.iloc[-1], 2))

        node_num = ['6', '9', '12', '16', '20', '24', '28', '32']

        fig = go.Figure(data=[
            go.Bar(name='WS', x=node_num, y=values_WS, text=values_WS, textposition='auto'),
            go.Bar(name='ER', x=node_num, y=values_ER, text=values_ER, textposition='auto'),
            go.Bar(name='BA', x=node_num, y=values_BA, text=values_BA, textposition='auto')
        ])
        # Change the bar mode
        fig.update_layout(
            title=f'{dataset} node number accuracy evaluation',
            title_x=0.5,
            xaxis_title="Number of nodes",
            yaxis_title="Accuracy [%]",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#000000"
            ),
            barmode='group'
        )
        fig.write_image(f"plot/{dataset}_node_accuracy_evaluation.pdf")
        fig.show()


def plot_empirical_time(dataset, seed):
    p = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
    k = np.arange(2, 11, 2)
    m = np.arange(1, 9, 1)
    er_p = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    values_WS = []
    p_x = []
    for prob in p:
        for node_ring in k:
            file_WS = pd.read_csv(
                f'./reporting/c_78_p_{prob}_graph_WS_dataset_{dataset}_seed_{seed}_name_empirical_one_opt_SGD_k_{node_ring}.csv')
            values_WS.append(round(file_WS.time.iloc[-1], 1))

            p_x.append(f'(p={prob},k={node_ring})')

    values_BA = []
    ba_x = []
    for num_nodes in m:
        file_BA = pd.read_csv(
            f'./reporting/c_78_p_0.75_graph_BA_dataset_{dataset}_seed_{seed}_name_empirical_one_opt_SGD_m_{num_nodes}.csv')
        values_BA.append(round(file_BA.time.iloc[-1], 1))
        ba_x.append(f'm={num_nodes}')

    values_ER = []
    er_x = []
    for prob in er_p:
        file_ER = pd.read_csv(
            f'./reporting/c_78_p_{prob}_graph_ER_dataset_{dataset}_seed_{seed}_name_empirical_one_opt_SGD.csv')
        values_ER.append(round(file_ER.time.iloc[-1], 1))
        er_x.append(f'p={prob}')

    fig = go.Figure(data=[
        go.Bar(name='WS', x=p_x, y=values_WS, text=values_WS, textposition='auto'),
        go.Bar(name='BA', x=ba_x, y=values_BA, text=values_BA, textposition='auto'),
        go.Bar(name='ER', x=er_x, y=values_ER, text=values_ER, textposition='auto')
    ])
    # Change the bar mode
    fig.update_layout(
        title=f'{dataset} network generator time performance',
        title_x=0.5,
        xaxis_title="Network generator parameters",
        yaxis_title="Time [s]",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#000000"
        ),
        barmode='group',
        width=1900, height=936
    )

    fig.write_image(f"plot/{dataset}_netgen_time_evaluation.pdf")
    fig.show()


def plot_empirical_accuracy(dataset, seed):
    p = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
    k = np.arange(2, 11, 2)
    m = np.arange(1, 9, 1)
    er_p = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    values_WS = []
    p_x = []
    for prob in p:
        for node_ring in k:
            file_WS = pd.read_csv(
                f'./reporting/c_78_p_{prob}_graph_WS_dataset_{dataset}_seed_{seed}_name_empirical_one_opt_SGD_k_{node_ring}.csv')
            values_WS.append(round(file_WS.accuracy.iloc[-1], 2))

            p_x.append(f'(p={prob},k={node_ring})')

    values_BA = []
    ba_x = []
    for num_nodes in m:
        file_BA = pd.read_csv(
            f'./reporting/c_78_p_0.75_graph_BA_dataset_{dataset}_seed_{seed}_name_empirical_one_opt_SGD_m_{num_nodes}.csv')
        values_BA.append(round(file_BA.accuracy.iloc[-1], 2))
        ba_x.append(f'm={num_nodes}')

    values_ER = []
    er_x = []
    for prob in er_p:
        file_ER = pd.read_csv(
            f'./reporting/c_78_p_{prob}_graph_ER_dataset_{dataset}_seed_{seed}_name_empirical_one_opt_SGD.csv')
        values_ER.append(round(file_ER.accuracy.iloc[-1], 2))
        er_x.append(f'p={prob}')

    fig = go.Figure(data=[
        go.Bar(name='WS', x=p_x, y=values_WS, text=values_WS, textposition='auto'),
        go.Bar(name='BA', x=ba_x, y=values_BA, text=values_BA, textposition='auto'),
        go.Bar(name='ER', x=er_x, y=values_ER, text=values_ER, textposition='auto')
    ])
    # Change the bar mode
    fig.update_layout(
        title=f'{dataset} network generator accuracy evaluation',
        title_x=0.5,
        xaxis_title="Network generator parameters",
        yaxis_title="Accuracy [%]",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#000000"
        ),
        barmode='group',
        width=1900, height=936
    )

    fig.write_image(f"plot/{dataset}_netgen_accuracy_evaluation.pdf")
    fig.show()


def plot_special():
    p = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]
    k = np.arange(2, 11, 2)

    values_WS = []
    p_x = []
    for prob in p:
        for node_ring in k:
            file_WS = pd.read_csv(
                f'./reporting/c_78_p_{prob}_graph_WS_dataset_CIFAR100_seed_6_name_empirical_one_opt_SGD_k_{node_ring}.csv')
            values_WS.append(round(file_WS.accuracy.iloc[-1], 2))

            p_x.append(f'(p={prob},k={node_ring})')

    fig = go.Figure(data=[
        go.Bar(name='WS', x=p_x, y=values_WS, text=values_WS, textposition='auto')
    ])
    # Change the bar mode
    fig.update_layout(
        title=f'CIFAR100 network generator accuracy evaluation',
        title_x=0.5,
        xaxis_title="Network generator parameters",
        yaxis_title="Accuracy [%]",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#000000"
        ),
        barmode='group',
        width=1900, height=936
    )

    fig.write_image(f"plot/CIFAR100_netgen_accuracy_evaluation.pdf")
    fig.show()

    values_WS = []
    p_x = []
    for prob in p:
        for node_ring in k:
            file_WS = pd.read_csv(
                f'./reporting/c_78_p_{prob}_graph_WS_dataset_CIFAR100_seed_6_name_empirical_one_opt_SGD_k_{node_ring}.csv')
            values_WS.append(round(file_WS.time.iloc[-1], 2))

            p_x.append(f'(p={prob},k={node_ring})')

    fig = go.Figure(data=[
        go.Bar(name='WS', x=p_x, y=values_WS, text=values_WS, textposition='auto')
    ])
    # Change the bar mode
    fig.update_layout(
        title_x=0.5,
        xaxis_title="Network generator parameters",
        yaxis_title="Time [s]",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#000000"
        ),
        barmode='group',
        width=1900, height=936
    )

    fig.write_image(f"plot/CIFAR100_netgen_time_evaluation.pdf")
    fig.show()


def main():
    current = ['CIFAR10', 'MNIST', 'FASHION_MNIST']

    for data in current:
        plot_empirical_time(data, 1)
        plot_empirical_accuracy(data, 1)

    plot_empirical_accuracy('CIFAR100', 6)
    plot_empirical_time('CIFAR100', 6)

    plot_node_study_time()
    plot_node_study_accuracy()

if __name__ == '__main__':
    main()
