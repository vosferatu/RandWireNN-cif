import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


def plot_seeds():
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    values_x = []

    for seed in seeds:
        file_data = pd.read_csv(f'./reporting/c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_{seed}_name_first_gen.csv')

        values_x.append(round(file_data.accuracy.iloc[-1], 2))

    fig = go.Figure(data=[
        go.Bar(x=seeds, y=values_x, text=values_x, textposition='auto'),
    ])
    # Change the bar mode
    fig.update_layout(
        title=f'Seed accuracy evaluation',
        title_x=0.5,
        xaxis_title="Seed",
        yaxis_title="Accuracy",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#000000"
        ),
        barmode='group',
        width=1900, height=936
    )

    fig.write_image(f"plot/seed_netgen_accuracy_evaluation.pdf")
    fig.show()


def plot_optimizer_evolution():
    train_data = pd.read_csv(f'./reporting/c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_-1_name_first_gen_opt_SGD0.csv')
    train_data = train_data[train_data['epoch'] < 51]
    sgd_default_data = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_-1_name_first_gen_opt_SGD1.csv')
    sgd_no_momentum_data = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_-1_name_first_gen_opt_SGD_NO_MOMENTUM1.csv')
    adam_data = pd.read_csv(
        f'./reporting/new_adam_c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_1_name_first_gen.csv')

    fig = go.Figure(data=[
        go.Scatter(x=train_data.epoch, y=train_data.accuracy, name='default weight training'),
        go.Scatter(x=sgd_default_data.epoch, y=sgd_default_data.accuracy, name='predicted_sgd_default'),
        go.Scatter(x=sgd_no_momentum_data.epoch, y=sgd_no_momentum_data.accuracy, name='predicted_sgd_no_momentum'),
        go.Scatter(x=adam_data.epoch, y=adam_data.accuracy, name='predicted_adam')
    ])
    # Change the bar mode
    fig.update_layout(
        legend=dict(
            x=0.7,
            y=0.2),
        title=f'Optimizer accuracy evaluation',
        title_x=0.5,
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#000000"
        ),
        barmode='group',
        width=1900, height=936
    )

    fig.write_image(f'./plot/first_gen_optimizer.pdf')
    fig.show()


def plot_frozen_experiments(dataset, network):
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

    fig = go.Figure(data=[
        go.Scatter(x=learned_weights.epoch, y=learned_weights.accuracy, name='default weight training'),
        go.Scatter(x=sgd_default_data.epoch, y=sgd_default_data.accuracy, name='predicted_sgd_default'),
        go.Scatter(x=sgd_frozen_ones.epoch, y=sgd_frozen_ones.accuracy, name='default weight initialization'),
        go.Scatter(x=sgd_frozen_dist.epoch, y=sgd_frozen_dist.accuracy, name='normal distribution'),
        go.Scatter(x=sgd_frozen_shuffled.epoch, y=sgd_frozen_shuffled.accuracy, name='shuffled weights')
    ])
    # Change the bar mode
    fig.update_layout(
        legend=dict(
            x=0.7,
            y=0.2),
        title=f'Connection weights experiments',
        title_x=0.5,
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#000000"
        ),
        barmode='group',
        width=1900, height=936
    )

    fig.write_image(f'./plot/weight_acc_focus_{dataset}_{network}.pdf')
    fig.show()


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


def final_acc_table(network):
    techniques = [['default_weight_training', 'predicted_sgd_default', 'ones', 'normal_dist', 'shuffled']]
    table_data = {'technique': techniques}

    for data in datasets:
        table_data[data] = get_final_acc(data, network)

    table_data = pd.DataFrame(table_data, columns=datasets, index=techniques)

    values = list(table_data)

    fig = go.Figure(data=[go.Table(
        header=dict(values=values,
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[table_data.MNIST, table_data.FASHION_MNIST, table_data.CIFAR10, table_data.CIFAR100],
                   fill_color='lavender',
                   align='left'))
    ])
    print(table_data)

    fig.show()


def plot_time_frozen_experiments(dataset, network):
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

    values = [round(learned_weights.time.iloc[-1], 1), round(sgd_default_data.time.iloc[-1], 1),
              round(sgd_frozen_ones.time.iloc[-1], 1),
              round(sgd_frozen_dist.time.iloc[-1], 1), round(sgd_frozen_shuffled.time.iloc[-1], 1)]

    legend = ['default training', 'predicted_sgd_default', 'default initialization', 'normal distribution',
              'shuffled']

    x = legend

    fig = go.Figure(data=[
        go.Bar(x=x, y=values, text=values, textposition='auto'),
    ])
    # Change the bar mode
    fig.update_layout(
        title=f'Time of connection weight experiments',
        title_x=0.5,
        xaxis_title="Technique",
        yaxis_title="Time",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#000000"
        ),
        barmode='group',
        width=1500, height=700

    )

    fig.write_image(f"plot/weight_time_focus_{dataset}_{network}.pdf")
    fig.show()


def plot_cooked_experiments(dataset, network):
    ax = plt.gca()

    learned_weights = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_frozen_2_opt_SGD0.csv')
    sgd_default_data = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_frozen_2_opt_SGD1.csv')
    sgd_frozen_ones = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_frozen_3_opt_SGDones.csv')
    sgd_pre_cooked = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_pre_cooked_opt_SGDpre_cooked.csv')
    sgd_pre_cooked_frozen = pd.read_csv(
        f'./reporting/c_78_p_0.75_graph_WS_dataset_{dataset}_seed_-1_name_pre_cooked_frozen_opt_SGDpre_cooked_frozen.csv')

    fig = go.Figure(data=[
        go.Scatter(x=learned_weights.epoch, y=learned_weights.accuracy, name='default weight training'),
        go.Scatter(x=sgd_default_data.epoch, y=sgd_default_data.accuracy, name='predicted_sgd_default'),
        go.Scatter(x=sgd_frozen_ones.epoch, y=sgd_frozen_ones.accuracy, name='default weight initialization'),
        go.Scatter(x=sgd_pre_cooked.epoch, y=sgd_pre_cooked.accuracy, name='\'cooked\' weights'),
        go.Scatter(x=sgd_pre_cooked_frozen.epoch, y=sgd_pre_cooked_frozen.accuracy,
                   name='\'cooked\' and frozen weights')
    ])
    # Change the bar mode
    fig.update_layout(
        legend=dict(
            x=0.7,
            y=0.05),
        title=f'\'Cooked\' weights experiments',
        title_x=0.5,
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#000000"
        ),
        barmode='group',
        width=1900, height=936
    )

    fig.write_image(f'./plot/weight_acc_cooked_{dataset}_{network}.pdf')
    fig.show()


def last():
    for data in datasets:
        plot_frozen_experiments(data, 'frozen_3')
        plot_time_frozen_experiments(data, 'frozen_3')


def main():
    # for data in datasets:
    #     plot_cooked_experiments(data, 'frozen')
    # last()
    # plot_optimizer_evolution()
    plot_seeds()

if __name__ == "__main__":
    main()
