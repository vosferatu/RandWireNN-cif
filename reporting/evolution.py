import pandas as pd
import matplotlib.pyplot as plt


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


for i in range(1, 11):
    plot_evolution(i)
