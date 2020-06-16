import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_evolution(seed):
    ax = plt.gca()

    train_data = pd.read_csv(f'./reporting/c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_{seed}_name_first_gen.csv')
    sgd_default_data = pd.read_csv(
        f'./reporting/new_sgd_default_c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_{seed}_name_first_gen.csv')
    sgd_no_momentum_data = pd.read_csv(
        f'./reporting/new_sgd_no_momentum_c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_{seed}_name_first_gen.csv')
    adam_data = pd.read_csv(f'./reporting/new_adam_c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_{seed}_name_first_gen.csv')

    train_data.plot(kind='line', x='epoch', y='accuracy', ax=ax, title='Accuracy evolution', legend=True)
    sgd_default_data.plot(kind='line', x='epoch', y='accuracy', ax=ax, title='Accuracy evolution', legend=True)
    sgd_no_momentum_data.plot(kind='line', x='epoch', y='accuracy', ax=ax, title='Accuracy evolution', legend=True)
    adam_data.plot(kind='line', x='epoch', y='accuracy', ax=ax, title='Accuracy evolution', legend=True)

    ax.legend(['random_weights', 'predicted_sgd_default', 'predicted_sgd_no_momentum', 'predicted_adam'])

    plt.savefig(f'./plot/first_gen_seed_{seed}.png')

    plt.clf()


def plot_frozen_experiments(dataset, network):
    ax = plt.gca()

    learned_weights = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_{network}_opt_SGDnormal.csv')
    sgd_default_data = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_{network}_opt_SGDpredict.csv')
    # sgd_no_momentum_data = pd.read_csv(
    #     f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_{network}_opt_SGD_NO_MOMENTUM1.csv')
    sgd_frozen_ones = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_{network}_opt_SGDones.csv')
    sgd_frozen_dist = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_{network}_opt_SGDdist.csv')
    sgd_frozen_shuffled = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_{network}_opt_SGDshuffled.csv')

    learned_weights.plot(
        kind='line', x='epoch', y='accuracy', ax=ax, legend=True)
    sgd_default_data.plot(
        kind='line', x='epoch', y='accuracy', ax=ax, legend=True)
    # sgd_no_momentum_data.plot(
    #     kind='line', x='epoch', y='accuracy', ax=ax, legend=True)
    sgd_frozen_ones.plot(
        kind='line', x='epoch', y='accuracy', ax=ax, legend=True)
    sgd_frozen_dist.plot(
        kind='line', x='epoch', y='accuracy', ax=ax, legend=True)
    sgd_frozen_shuffled.plot(
        kind='line', x='epoch', y='accuracy', ax=ax, legend=True)

    ax.set_title(f'Accuracy evolution {dataset} - {network}')

    ax.legend(['learned_1_weights', 'predicted_sgd_default', 'ones', 'normal_dist', 'shuffled'])

    plt.savefig(f'./plot/{network}_{dataset}.png')
    plt.clf()


datasets = ['MNIST', 'FASHION_MNIST', 'CIFAR10', 'CIFAR100']
optimizers = ['SGD0', 'SGD1', 'SGD_NO_MOMENTUM1']


def get_final_acc(dataset, network):
    ax = plt.gca()

    learned_weights = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_{network}_opt_SGDnormal.csv')
    sgd_default_data = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_{network}_opt_SGDpredict.csv')
    # sgd_no_momentum_data = pd.read_csv(
    #     f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_{network}_opt_SGD_NO_MOMENTUM1.csv')
    sgd_frozen_ones = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_{network}_opt_SGDones.csv')
    sgd_frozen_dist = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_{network}_opt_SGDdist.csv')
    sgd_frozen_shuffled = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_{network}_opt_SGDshuffled.csv')

    max_learned = learned_weights.accuracy.iloc[-1]
    max_sgd = sgd_default_data.accuracy.iloc[-1]
    max_frozen_ones = sgd_frozen_ones.accuracy.iloc[-1]
    max_frozen_dist = sgd_frozen_dist.accuracy.iloc[-1]
    max_frozen_shuffled = sgd_frozen_shuffled.accuracy.iloc[-1]

    return [max_learned, max_sgd, max_frozen_ones, max_frozen_dist, max_frozen_shuffled]


def final_acc_table():
    techniques = ['learned_1_weights', 'predicted_sgd_default', 'ones', 'normal_dist', 'shuffled']
    table_data = {'technique': techniques}

    for data in datasets:
        table_data[data] = get_final_acc(data, 'frozen_3')

    table_data = pd.DataFrame(table_data, columns=datasets, index=techniques)

    print(table_data)


def plot_time_frozen_experiments(dataset, network):
    ax = plt.gca()

    learned_weights = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_{network}_opt_SGDnormal.csv')
    sgd_default_data = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_{network}_opt_SGDpredict.csv')
    # sgd_no_momentum_data = pd.read_csv(
    #     f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_{network}_opt_SGD_NO_MOMENTUM1.csv')
    sgd_frozen_ones = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_{network}_opt_SGDones.csv')
    sgd_frozen_dist = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_{network}_opt_SGDdist.csv')
    sgd_frozen_shuffled = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_{network}_opt_SGDshuffled.csv')

    values = [learned_weights.time.iloc[-1], sgd_default_data.time.iloc[-1], sgd_frozen_ones.time.iloc[-1],
              sgd_frozen_dist.time.iloc[-1], sgd_frozen_shuffled.time.iloc[-1]]

    ax.legend(['learned_1_weights', 'predicted_sgd_default', 'ones', 'normal_dist', 'shuffled'])

    x = np.arange(5)

    plt.bar(x, values)
    plt.xticks(x, ('learned_1_weights', 'predicted_sgd_default', 'ones', 'normal_dist', 'shuffled'))

    plt.title(dataset)

    plt.savefig(f'./plot/{network}_time_{dataset}.png')
    plt.clf()


def plot_cooked_experiments(dataset, network):
    ax = plt.gca()

    learned_weights = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_frozen_2_opt_SGD0.csv')
    sgd_default_data = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_frozen_2_opt_SGD1.csv')
    # sgd_no_momentum_data = pd.read_csv(
    #     f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_frozen_2_opt_SGD_NO_MOMENTUM1.csv')
    sgd_frozen_ones = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_frozen_2_opt_SGDones.csv')
    sgd_frozen_dist = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_frozen_2_opt_SGDdist.csv')
    sgd_pre_cooked = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_pre_cooked_opt_SGDpre_cooked.csv')
    sgd_pre_cooked_frozen = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_{network}_opt_SGDpre_cooked_frozen.csv')

    learned_weights.plot(
        kind='line', x='epoch', y='accuracy', ax=ax, legend=True)
    sgd_default_data.plot(
        kind='line', x='epoch', y='accuracy', ax=ax, legend=True)
    # sgd_no_momentum_data.plot(
    #     kind='line', x='epoch', y='accuracy', ax=ax, title=f'Accuracy evolution {dataset} - {network}', legend=True)
    sgd_frozen_ones.plot(
        kind='line', x='epoch', y='accuracy', ax=ax, legend=True)
    sgd_frozen_dist.plot(
        kind='line', x='epoch', y='accuracy', ax=ax, legend=True)
    sgd_pre_cooked.plot(
        kind='line', x='epoch', y='accuracy', ax=ax, legend=True)
    sgd_pre_cooked_frozen.plot(
        kind='line', x='epoch', y='accuracy', ax=ax, legend=True)

    ax.set_title(f'Accuracy evolution {dataset} - {network}')

    ax.legend(['learned_1_weights', 'predicted_sgd_default', 'ones', 'normal_dist', 'pre_cooked', 'pre_cooked_frozen'])

    plt.savefig(f'./plot/pre_cooked_frozen_{dataset}.png')
    plt.clf()


def main():
    for data in datasets:
        plot_frozen_experiments(data, 'frozen_3')
        plot_time_frozen_experiments(data, 'frozen_3')

    final_acc_table()


if __name__ == "__main__":
    main()
