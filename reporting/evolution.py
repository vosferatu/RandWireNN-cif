import matplotlib.pyplot as plt
import pandas as pd


def plot_evolution(seed):
    ax = plt.gca()

    train_data = pd.read_csv(f'c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_{seed}_name_first_gen.csv')
    sgd_default_data = pd.read_csv(f'new_sgd_default_c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_{seed}_name_first_gen.csv')
    sgd_no_momentum_data = pd.read_csv(f'new_sgd_no_momentum_c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_{seed}_name_first_gen.csv')
    adam_data = pd.read_csv(f'new_adam_c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_{seed}_name_first_gen.csv')

    train_data.plot(kind='line', x='epoch', y='accuracy', ax=ax, title='Accuracy evolution', legend=True)
    sgd_default_data.plot(kind='line', x='epoch', y='accuracy', ax=ax, title='Accuracy evolution', legend=True)
    sgd_no_momentum_data.plot(kind='line', x='epoch', y='accuracy', ax=ax, title='Accuracy evolution', legend=True)
    adam_data.plot(kind='line', x='epoch', y='accuracy', ax=ax, title='Accuracy evolution', legend=True)

    ax.legend(['random_weights', 'predicted_sgd_default', 'predicted_sgd_no_momentum', 'predicted_adam'])

    plt.savefig(f'./plots/first_gen_seed_{seed}.png')

    plt.clf()


def plot_dataset(dataset):
    ax = plt.gca()

    random_weights = pd.read_csv(f'c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_first_gen_opt_SGD0.csv')
    sgd_default_data = pd.read_csv(f'c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_first_gen_opt_SGD1.csv')
    sgd_no_momentum_data = pd.read_csv(
        f'c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_first_gen_opt_SGD_NO_MOMENTUM1.csv')

    random_weights.plot(kind='line', x='epoch', y='accuracy', ax=ax, title=f'Accuracy evolution {dataset}', legend=True)
    sgd_default_data.plot(kind='line', x='epoch', y='accuracy', ax=ax, title=f'Accuracy evolution {dataset}',
                          legend=True)
    sgd_no_momentum_data.plot(kind='line', x='epoch', y='accuracy', ax=ax, title=f'Accuracy evolution {dataset}',
                              legend=True)

    ax.legend(['random_weights', 'predicted_sgd_default', 'predicted_sgd_no_momentum'])

    plt.savefig(f'./plots/first_gen_{dataset}.png')

    plt.clf()


def plot_optimizer(opt):
    ax = plt.gca()

    mnist = pd.read_csv(f'c_78_p_0.75_graph_WS_dataset_MNIST_seed_-1_name_first_gen_opt_{opt}.csv')
    fashion_mnist = pd.read_csv(f'c_78_p_0.75_graph_WS_dataset_FASHION_MNIST_seed_-1_name_first_gen_opt_{opt}.csv')
    cifar10 = pd.read_csv(f'c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_-1_name_first_gen_opt_{opt}.csv')
    cifar100 = pd.read_csv(f'c_78_p_0.75_graph_WS_dataset_CIFAR100_seed_-1_name_first_gen_opt_{opt}.csv')

    mnist.plot(kind='line', x='epoch', y='accuracy', ax=ax, title=f'Accuracy evolution {opt}', legend=True)
    fashion_mnist.plot(kind='line', x='epoch', y='accuracy', ax=ax, title=f'Accuracy evolution {opt}', legend=True)
    cifar10.plot(kind='line', x='epoch', y='accuracy', ax=ax, title=f'Accuracy evolution {opt}', legend=True)
    cifar100.plot(kind='line', x='epoch', y='accuracy', ax=ax, title=f'Accuracy evolution {opt}', legend=True)

    ax.legend(['mnist', 'fashion_mnist', 'cifar10', 'cifar100'])

    plt.savefig(f'./plots/first_gen_{opt}.png')

    plt.clf()


datasets = ['MNIST', 'FASHION_MNIST', 'CIFAR10', 'CIFAR100']
optimizers = ['SGD0', 'SGD0', 'SGD_NO_MOMENTUM1']

for i in datasets:
    plot_dataset(i)

for i in optimizers:
    plot_optimizer(i)
