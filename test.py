import random
import statistics as st

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from numpy.random import RandomState
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import train_utils
from node_data import forest_run

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

datasets = ['CIFAR10', 'CIFAR100']


def evaluate_forest_first_gen(dataset='CIFAR10', name='first_gen'):
    for seed in range(1, 11):
        model, _, _ = train_utils.load_model(train_utils.Args(dataset, seed, name))

        df = train_utils.get_edge_dataset(model)
        df_new = forest_run.predict_weights(df)

        x = df.weight
        y = df_new.weight

        plt.plot(x, y, '.')

        plt.savefig(f'./weights/{dataset}_{seed}.png')
        plt.clf()


def plot_pred_vs_true(dataset='CIFAR10', name='first_gen'):
    for seed in range(1, 11):
        model, _, _ = train_utils.load_model(train_utils.Args(dataset, seed, name))

        df = train_utils.get_edge_dataset(model)
        df_new = forest_run.predict_weights(df)

        y_true = df.weight
        y_pred = df_new.weight

        plt.plot(y_true)
        plt.plot(y_pred)
        plt.legend(["true", "pred"])

        plt.savefig(f'./weights/{dataset}_{seed}_2.png')
        plt.clf()


def shuffle_weights(df):
    weights = df.weight
    prev_len = len(weights)
    weights = weights[weights != 1.0]
    rep = [1.0] * (prev_len - len(weights))
    simple = weights.tolist()
    np.random.shuffle(simple)

    ones = pd.Series(rep)
    weights = pd.Series(simple)
    weights = weights.append(ones, ignore_index=True)

    df.weight = weights
    return df


def randomize_weights(df):
    weights = df.weight
    prev_len = len(weights)
    weights = weights[weights != 1.0]
    rep = [1.0] * (prev_len - len(weights))
    simple = weights.tolist()

    simple = np.random.uniform(min(simple), max(simple), len(simple))

    ones = pd.Series(rep)
    weights = pd.Series(simple)
    weights = weights.append(ones, ignore_index=True)

    df.weight = weights
    return df


def evaluate_predict_shuffle_end(rand=0):
    for dataset in datasets:
        model, _, test_loader = train_utils.new_model(train_utils.Args(dataset, -1, 'potatoes', True))
        random_acc = train_utils.get_test(model, test_loader)
        print('random_acc: ', random_acc)
        predicted_acc, data = train_utils.run_predicted_eval(model, test_loader)
        print('predicted_acc: ', predicted_acc)

        shuffled_acc = []
        for i in range(0, 100):
            np.random.seed()
            if rand:
                data = randomize_weights(data)
            else:
                data = shuffle_weights(data)
            train_utils.adjust_edge_weights(model, data)
            new_acc = train_utils.get_test(model, test_loader)
            shuffled_acc.append(new_acc)
            print('shuffled no. ', i, ': ', new_acc)

        st_dev = st.pstdev(shuffled_acc)
        mean = st.mean(shuffled_acc)

        fig = go.Figure(data=[
            go.Bar(name='default model', x=['random'], y=[random_acc]),
            go.Bar(name='predicted', x=['pred'], y=[predicted_acc]),
            go.Bar(name='shuffle', x=['shuffled'], y=[mean], error_y=[st_dev])
        ])
        # Change the bar mode
        fig.update_layout(
            title=f'Untrained evaluation',
            title_x=0.5,
            xaxis_title="Technique",
            yaxis_title="Accuracy",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#000000"
            ),
            barmode='group',
            width=1500, height=700
        )

        fig.write_image(f"weights/shuffled_{dataset}_untrained_100.pdf")
        fig.show()


def normal_distribution_weights(model, train_weights=True):
    first_gen_df = pd.read_csv('./node_data/first_gen.csv')
    first_gen_weights = first_gen_df.weight.tolist()

    df = train_utils.get_edge_dataset(model)

    weights = df.weight
    prev_len = len(weights)
    weights = weights[weights != 1.0]
    rep = [1.0] * (prev_len - len(weights))
    simple = weights.tolist()

    st_dev = st.pstdev(first_gen_weights)
    mean = st.mean(first_gen_weights)

    np.random.seed()
    simple = np.random.normal(mean, st_dev, len(simple))

    ones = pd.Series(rep)
    weights = pd.Series(simple)
    weights = weights.append(ones, ignore_index=True)

    df.weight = weights

    train_utils.adjust_edge_weights(model, df, train_weights)


def run_frozen_predict_connections():
    for dataset in datasets:
        args = train_utils.Args(dataset, -1, 'frozen_3')
        model, train_loader, test_loader = train_utils.load_model(args, '_init_')
        train_utils.run_predicted_eval(model, test_loader)

        train_utils.freeze_weights(model)

        train_utils.run_epochs(model, args, train_loader, test_loader, 'predict')


def run_frozen_predict_ones():
    for dataset in datasets:
        args = train_utils.Args(dataset, -1, 'frozen_3')
        model, train_loader, test_loader = train_utils.load_model(args, '_init_')

        train_utils.freeze_weights(model)

        train_utils.run_epochs(model, args, train_loader, test_loader, 'ones')


def run_frozen_normal_distribution():
    for dataset in datasets:
        args = train_utils.Args(dataset, -1, 'frozen_3')
        model, train_loader, test_loader = train_utils.load_model(args, '_init_')

        normal_distribution_weights(model, False)

        train_utils.run_epochs(model, args, train_loader, test_loader, 'dist')


def hist_weights():
    first_gen_df = pd.read_csv('./node_data/c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_1_name_first_gen.csv')

    weights = first_gen_df.weight

    first_gen_weights = first_gen_df.weight.tolist()

    st_dev = st.pstdev(first_gen_weights)
    mean = st.mean(first_gen_weights)

    print('st_dev: ', st_dev)
    print('mean: ', mean)

    # ax = weights.plot.hist(bins=30, alpha=0.5)

    fig = go.Figure(data=[
        go.Histogram(x=weights)
    ])
    # Change the bar mode
    fig.update_layout(
        title_x=0.5,
        xaxis_title="Weight",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#000000"
        ),
        barmode='group'
    )

    fig.write_image(f"plot/weight_distribution.pdf")
    fig.show()


def get_cooked_model():
    for dataset in datasets:
        args = train_utils.Args(dataset, -1, 'frozen_3', True)
        model, train_loader, test_loader = train_utils.new_model(args)
        train_utils.run_predicted_eval(model, test_loader)

        train_utils.cook_model(model)

        train_utils.freeze_nodes(model)

        train_utils.run_epochs(model, args, train_loader, test_loader, 'pre_cooked_frozen')


def run_normal():
    for dataset in datasets:
        args = train_utils.Args(dataset, -1, 'frozen_3', True)
        model, train_loader, test_loader = train_utils.new_model(args)

        train_utils.run_epochs(model, args, train_loader, test_loader, 'normal')


def run_shuffled():
    for dataset in datasets:
        args = train_utils.Args(dataset, -1, 'frozen_3')
        model, train_loader, test_loader = train_utils.load_model(args, '_init_')

        predicted_acc, data = train_utils.run_predicted_eval(model, test_loader)
        print('predicted_acc: ', predicted_acc)
        data = shuffle_weights(data)
        train_utils.adjust_edge_weights(model, data)

        train_utils.freeze_weights(model)
        train_utils.run_epochs(model, args, train_loader, test_loader, 'shuffled')


def cluster(dataset):
    args = train_utils.Args(dataset, -1, 'frozen_3')
    model, _, _ = train_utils.load_model(args, '_end_')

    df = train_utils.get_kernel_dataset(model)

    df.drop(["channel1", "channel2", "node"], axis=1, inplace=True)
    df = df[df.level == 1]
    df.drop(["level"], axis=1, inplace=True)

    dataset_metrics = df.describe()
    print(dataset_metrics)
    means = dataset_metrics.iloc[2].tolist()
    print('mean:', st.mean(means))

    stscaler = StandardScaler().fit(df)
    df = stscaler.transform(df)
    df = stscaler.inverse_transform(df)

    print(df)

    cluster_model = DBSCAN(eps=0.05, min_samples=3).fit(df)

    core_samples_mask = np.zeros_like(cluster_model.labels_, dtype=bool)
    core_samples_mask[cluster_model.core_sample_indices_] = True
    labels = cluster_model.labels_

    print('num_labels: ', len(labels))
    print('labels: ', labels)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = df[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = df[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title(f'Estimated number of clusters: %d - {dataset}' % n_clusters_)

    plt.savefig(f'./plot/cluster_frozen_3_{dataset}.png')
    plt.clf()


def kernel_dataset():
    dfs = []

    for i in range(1, 11):
        args = train_utils.Args('CIFAR10', i, 'first_gen')
        model, _, _ = train_utils.load_model(args)

        dfs.append(train_utils.get_kernel_dataset(model))

    df = pd.concat(dfs, ignore_index=True)

    df.to_csv('./node_data/first_gen_kernels.csv')


def empirical():
    num_experiment = 0

    np.random.seed()
    seed = random.randint(1, 10)

    p = np.arange(0.0, 1.0, 0.15)
    p = np.append(p, [1.0])
    k = np.arange(2, 11, 2)
    m = np.arange(1, 9, 1)
    er_p = np.arange(0.0, 0.9, 0.1)

    print(p)
    print(k)
    print(m)
    print(er_p)

    for dataset in datasets:

        for prob in p:
            for node_ring in k:
                args = train_utils.Args(dataset, seed, 'empirical_one', True)
                args.p = prob
                args.k = node_ring
                model, train_loader, test_loader = train_utils.new_model(args)

                train_utils.run_epochs(model, args, train_loader, test_loader, '_k_' + str(args.k))
                num_experiment = num_experiment + 1
                print('experiment: ', num_experiment)

        for num_nodes in m:
            args = train_utils.Args(dataset, seed, 'empirical_one', True)
            args.m = num_nodes
            args.graph_mode = "BA"
            model, train_loader, test_loader = train_utils.new_model(args)

            train_utils.run_epochs(model, args, train_loader, test_loader, '_m_' + str(args.m))

            num_experiment = num_experiment + 1
            print('experiment: ', num_experiment)

        for prob in er_p:
            args = train_utils.Args(dataset, seed, 'empirical_one', True)
            args.p = prob
            args.graph_mode = "ER"
            model, train_loader, test_loader = train_utils.new_model(args)

            train_utils.run_epochs(model, args, train_loader, test_loader)
            num_experiment = num_experiment + 1
            print('experiment: ', num_experiment)


def imagenet_tryout():
    args = train_utils.Args('IMAGENET', 1, 'empirical_two', True)
    args.p = 0.2
    args.graph_mode = 'ER'
    args.node_num = 6
    args.epochs = 250
    args.batch_size = 32

    model, train_loader, test_loader = train_utils.new_model(args)

    # normal run

    train_utils.run_epochs(model, args, train_loader, test_loader)

    # prediction and frozen

    model, train_loader, test_loader = train_utils.new_model(args)

    train_utils.run_predicted_eval(model, test_loader)

    train_utils.freeze_weights(model)

    train_utils.run_epochs(model, args, train_loader, test_loader, '_frozen')


def node_testing():
    num_experiment = 1
    nodes_check = [6, 9, 12, 16, 20, 24, 28]

    for dataset in datasets:
        for node_num in nodes_check:
            args = train_utils.Args(dataset, 1, 'empirical_nodes', True)
            args.node_num = node_num
            args.graph_mode = "WS"
            model, train_loader, test_loader = train_utils.new_model(args)

            train_utils.run_epochs(model, args, train_loader, test_loader, '_nodes_' + str(args.node_num))

            num_experiment = num_experiment + 1
            print('experiment: ', num_experiment)

            args = train_utils.Args(dataset, 1, 'empirical_nodes', True)
            args.node_num = node_num
            args.graph_mode = "BA"
            args.m = 5
            model, train_loader, test_loader = train_utils.new_model(args)

            train_utils.run_epochs(model, args, train_loader, test_loader, '_nodes_' + str(args.node_num))

            num_experiment = num_experiment + 1
            print('experiment: ', num_experiment)

            args = train_utils.Args(dataset, 1, 'empirical_nodes', True)
            args.node_num = node_num
            args.graph_mode = "ER"
            args.p = 0.2
            model, train_loader, test_loader = train_utils.new_model(args)

            train_utils.run_epochs(model, args, train_loader, test_loader, '_nodes_' + str(args.node_num))

            num_experiment = num_experiment + 1
            print('experiment: ', num_experiment)


def main():
    evaluate_predict_shuffle_end()
    # hist_weights()
    # imagenet_tryout()


if __name__ == "__main__":
    main()
