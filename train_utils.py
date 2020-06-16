import argparse
import collections
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import node_data.forest_run as forest
import randwire
from model import Model
from plot import draw_plot
from preprocess import load_data

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 2, T_mult=2)


class Args:
    def __init__(self, dataset, seed, name, is_train=False):
        self.dataset_mode = dataset
        self.seed = seed
        self.name = name
        self.is_train = is_train

    p = 0.75
    c = 78
    k = 4
    m = 5
    graph_mode = "WS"
    node_num = 32
    learning_rate = 1e-1
    batch_size = 128
    model_mode = "SMALL_REGIME"
    optimizer = 'SGD'
    epochs = 50


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="evaluation", mininterval=1):
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(target.data).sum()

    acc = 100. * float(correct) / len(test_loader.dataset)
    return acc


def run_predicted_eval(model, test_loader):
    df = get_edge_dataset(model)
    new_weights = forest.predict_weights(df)
    adjust_edge_weights(model, new_weights)

    acc = get_test(model, test_loader)
    print('predicted_weight accuracy: {0:.2f}%'.format(acc))
    return acc, new_weights


def run_direct_eval(model, test_loader):
    acc = get_test(model, test_loader)
    print('random_weight accuracy: {0:.2f}%'.format(acc))


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for value in list(layer.size()):
            layer_parameter *= value
        total_parameters += layer_parameter
    return total_parameters


def check_weights(model, args, text):
    with open("./weights/" + "dataset_" + args.dataset_mode + "_seed_" + str(args.seed) + "_name_" + args.name
              + text + ".txt", "a") as f:
        f.write("_opt_" + args.optimizer + str(args.opt) + '\n')
        for graph_conv in model.children():
            if type(graph_conv[0]) == randwire.RandWire:

                for node in graph_conv[0].module_list:
                    if len(node.in_edges) > 1:
                        f.write(str(node.node) + '-' + str(node.in_edges) + '\n')
                        f.write(str(node.weights) + '\n')
                    else:
                        f.write(str(node.node) + '-' + str(node.in_edges) + '\n')
                        f.write('node_op\n')


def adjust_edge_weights(model, data, train_weights=True):
    curr_level = 0

    for graph_conv in model.children():
        if type(graph_conv[0]) == randwire.RandWire:
            curr_level = curr_level + 1
            node_weights = collections.defaultdict(list)
            for i in data.itertuples():
                if i.level == curr_level:
                    nodes = i.connection.split('-')
                    node_a = int(nodes[0])
                    node_b = int(nodes[1])
                    node_weights[node_b].append(i.weight)

            for node in graph_conv[0].module_list:
                if len(node.in_edges) > 1:
                    node.weights = torch.nn.Parameter(torch.cuda.FloatTensor(node_weights[node.node]),
                                                      requires_grad=train_weights)


def adjust_seed_weights(model, seed, train_weights=True):
    data = pd.read_csv(f'./node_data/dataset_{seed}.csv')
    new_weights = forest.predict_weights(data)
    adjust_edge_weights(model, new_weights)


def adjust_node_weights(model, data, train_weights=True):
    curr_level = 0

    for graph_conv in model.children():
        if type(graph_conv[0]) == randwire.RandWire:
            curr_level = curr_level + 1
            node_weights = collections.defaultdict(list)
            for i in data.itertuples():
                if i.level == curr_level:
                    node_weights[data.node].append(i.weight)


def get_edge_dataset(model):
    connection = []
    weight = []
    level = []
    curr_level = 0
    in_degree = []
    out_degree = []
    graph_metrics = {}

    for graph_conv in model.children():
        if type(graph_conv[0]) == randwire.RandWire:
            curr_level = curr_level + 1
            _in_degree = graph_conv[0].graph.get_in_degree()
            _out_degree = graph_conv[0].graph.get_out_degree()
            curr_edges = []
            metrics = {}
            for node in graph_conv[0].module_list:
                if len(node.in_edges) > 1:
                    for y in range(len(node.in_edges)):
                        curr_edges.append(str(node.in_edges[y]) + '-' + str(node.node))
                        weight.append(node.weights[y].item())
                        level.append(curr_level)
                        in_degree.append(_in_degree[node.in_edges[y]])
                        out_degree.append((_out_degree[node.node]))

            graph_conv[0].graph.get_edge_metrics(metrics, curr_edges)

            if not graph_metrics:
                graph_metrics.update(metrics)
            else:
                for x in graph_metrics:
                    graph_metrics[x].extend(metrics[x])

            connection.extend(curr_edges)

    seed = [model.seed] * len(connection)

    d = {'seed': seed, 'connection': connection, 'weight': weight, 'level': level, 'in_degree': in_degree,
         'out_degree': out_degree}
    d.update(graph_metrics)

    df = pd.DataFrame(data=d)
    print(df)
    return df


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def freeze_weights(model):
    for graph_conv in model.children():
        if type(graph_conv[0]) == randwire.RandWire:
            for node in graph_conv[0].module_list:
                if len(node.in_edges) > 1:
                    node.weights.requires_grad = False
                # for op in node.unit.children():
                #     op[1].conv.weight.requires_grad = False
                #     op[1].conv.pointwise.requires_grad = False


def get_node_dataset(model):
    total_nodes = []
    weight = []
    multi_layer = []
    level = []
    curr_level = 0
    in_degree = []
    out_degree = []
    pos_i = []
    pos_j = []
    graph_node_metrics = {}

    for graph_conv in model.children():
        if type(graph_conv[0]) == randwire.RandWire:
            curr_level = curr_level + 1
            _in_degree = graph_conv[0].graph.get_in_degree()
            _out_degree = graph_conv[0].graph.get_out_degree()
            curr_nodes = []
            metrics = {}
            for node in graph_conv[0].module_list:
                for op in node.unit.children():
                    for i in range(0, len(op[1].conv.weight[0][0])):
                        for j in range(0, len(op[1].conv.weight[0][0][i])):
                            curr_nodes.append(node.node)
                            level.append(curr_level)
                            in_degree.append(_in_degree[node.node])
                            out_degree.append((_out_degree[node.node]))
                            weight.append(op[1].conv.weight[0][0][i][j].item())
                            multi_layer.append(1)
                            pos_i.append(i)
                            pos_j.append(j)

                curr_nodes.append(node.node)
                level.append(curr_level)
                in_degree.append(_in_degree[node.node])
                out_degree.append((_out_degree[node.node]))
                weight.append(op[1].pointwise.weight[0][0][0].item())
                multi_layer.append(0)

            graph_conv[0].graph.get_node_metrics(metrics, curr_nodes)

            if not graph_node_metrics:
                graph_node_metrics.update(metrics)
            else:
                for x in graph_node_metrics:
                    graph_node_metrics[x].extend(metrics[x])

            total_nodes.extend(curr_nodes)

    seed = [model.seed] * len(node)

    d = {'seed': seed, 'node': total_nodes, 'weight': weight, 'level': level, 'multi_layer': multi_layer,
         'in_degree': in_degree, 'out_degree': out_degree, 'pos_i': pos_i, 'pos_j': pos_j}
    d.update(graph_node_metrics)

    df = pd.DataFrame(data=d)
    print(df)
    return df


def load_model(args, text=''):
    train_loader, test_loader = load_data(args)

    if os.path.exists("./checkpoint"):
        model = Model(args.node_num, args.p, args.c, args.c, args.graph_mode, args.model_mode, args.dataset_mode,
                      args.is_train, args.k, args.m, args.name + '_' + args.dataset_mode, args.seed).to(device)
        filename = "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_" + args.graph_mode + "_dataset_" + \
                   args.dataset_mode + "_seed_" + str(args.seed) + "_name_" + args.name
        print(filename)
        checkpoint = torch.load('./checkpoint/' + filename + text + 'ckpt.t7')
        model.load_state_dict(checkpoint['model'], strict=False)
        end_epoch = checkpoint['epoch']
        best_acc = checkpoint['acc']
        print("[Saved Best Accuracy]: ", best_acc, '%', "[End epochs]: ", end_epoch)
        # print("Number of model parameters: ", get_model_parameters(model))

        if device == 'cuda':
            model = torch.nn.DataParallel(model)

        return model, train_loader, test_loader

    else:
        assert os.path.exists("./checkpoint/" + str(args.seed) + "ckpt.t7"), "File not found. Please check again."


def save_model(model, args, text='_end_'):
    _, test_loader = load_data(args)

    state = {
        'model': model.state_dict(),
        'acc': get_test(model, test_loader),
        'epoch': 0,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    filename = "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_" + args.graph_mode + "_dataset_" + \
               args.dataset_mode + "_seed_" + str(args.seed) + "_name_" + args.name
    torch.save(state, './checkpoint/' + filename + text + 'ckpt.t7')


def train(model, train_loader, optimizer, criterion, epoch, args):
    model.train()
    step = 0
    train_loss = 0
    train_acc = 0
    for data, target in tqdm(train_loader, desc="epoch " + str(epoch), mininterval=1):
        adjust_learning_rate(optimizer, epoch, args)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        y_pred = output.data.max(1)[1]

        acc = float(y_pred.eq(target.data).sum()) / len(data) * 100.
        train_acc += acc
        step += 1
        # scheduler.step(epoch + step / args.batch_size)

        if step % 100 == 0:
            print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epoch, loss.data, acc), end='')
            for param_group in optimizer.param_groups:
                print(",  Current learning rate is: {}".format(param_group['lr']))

    length = len(train_loader.dataset) // args.batch_size
    return train_loss / length, train_acc / length


def new_model(args):
    train_loader, test_loader = load_data(args)

    model = Model(args.node_num, args.p, args.c, args.c, args.graph_mode, args.model_mode, args.dataset_mode,
                  args.is_train, args.k, args.m, args.name + '_' + args.dataset_mode, args.seed).to(device)
    state = {
        'model': model.state_dict(),
        'epoch': 0,
        'acc': run_direct_eval(model, test_loader)
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    filename = "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_" + args.graph_mode + "_dataset_" + \
               args.dataset_mode + "_seed_" + str(args.seed) + "_name_" + args.name
    torch.save(state, './checkpoint/' + filename + '_init_ckpt.t7')

    return model, train_loader, test_loader


def get_cooked_separable_conv(level, size):
    np.random.seed()
    seed = np.random.randint(low=1, high=11)
    args = Args('CIFAR10', seed, 'first_gen', False)
    model, _, _ = load_model(args)
    node_to_copy = np.random.randint(low=0, high=34)
    curr_level = 2

    for graph_conv in model.children():
        if type(graph_conv[0]) == randwire.RandWire:
            curr_level = curr_level + 1
            if curr_level == level:
                while graph_conv[0].module_list[node_to_copy].unit.unit[1].conv.weight.size() != size:
                    node_to_copy = np.random.randint(low=0, high=34)

                return graph_conv[0].module_list[node_to_copy].unit.unit[1]


def cook_model(model):
    curr_level = 2

    for graph_conv in model.children():
        if type(graph_conv[0]) == randwire.RandWire:
            curr_level = curr_level + 1
            for node in graph_conv[0].module_list:
                cooked_node = get_cooked_separable_conv(curr_level, node.unit.unit[1].conv.weight.size())

                node.unit.unit[1].conv.weight = torch.nn.Parameter(cooked_node.conv.weight.detach().requires_grad_())
                node.unit.unit[1].conv.bias = torch.nn.Parameter(cooked_node.conv.bias.detach().requires_grad_())
                node.unit.unit[1].pointwise.weight = torch.nn.Parameter(cooked_node.pointwise.weight.detach().
                                                                        requires_grad_())
                node.unit.unit[1].pointwise.bias = torch.nn.Parameter(cooked_node.pointwise.bias.detach().
                                                                      requires_grad_())

    model.to(device)


def freeze_nodes(model):
    for graph_conv in model.children():
        if type(graph_conv[0]) == randwire.RandWire:
            for node in graph_conv[0].module_list:
                node.unit.unit[1].conv.weight.requires_grad = False
                node.unit.unit[1].conv.bias.requires_grad = False
                node.unit.unit[1].pointwise.weight.requires_grad = False
                node.unit.unit[1].pointwise.bias.requires_grad = False


def get_kernel_dataset(model):
    curr_level = 0
    level = []
    nodes = []
    channel1 = []
    channel2 = []
    weight9 = []
    weight1 = []
    weight2 = []
    weight3 = []
    weight4 = []
    weight5 = []
    weight6 = []
    weight7 = []
    weight8 = []
    bias = []

    graph_node_metrics = {}

    for graph_conv in model.children():
        if type(graph_conv[0]) == randwire.RandWire:
            curr_level = curr_level + 1
            curr_nodes = []
            metrics = {}
            for node in graph_conv[0].module_list:
                for op in node.unit.children():
                    for i in range(0, len(op[1].conv.weight)):
                        for j in range(0, len(op[1].conv.weight[i])):
                            curr_nodes.append(node.node)
                            level.append(curr_level)
                            channel1.append(i)
                            channel2.append(j)
                            weight1.append(op[1].conv.weight[i][j][0][0].item())
                            weight2.append(op[1].conv.weight[i][j][0][1].item())
                            weight3.append(op[1].conv.weight[i][j][0][2].item())
                            weight4.append(op[1].conv.weight[i][j][1][0].item())
                            weight5.append(op[1].conv.weight[i][j][1][1].item())
                            weight6.append(op[1].conv.weight[i][j][1][2].item())
                            weight7.append(op[1].conv.weight[i][j][2][0].item())
                            weight8.append(op[1].conv.weight[i][j][2][1].item())
                            weight9.append(op[1].conv.weight[i][j][2][2].item())
                            bias.append(op[1].conv.bias[i].item())

            graph_conv[0].graph.get_node_metrics(metrics, curr_nodes)

            if not graph_node_metrics:
                graph_node_metrics.update(metrics)
            else:
                for x in graph_node_metrics:
                    graph_node_metrics[x].extend(metrics[x])

            nodes.extend(curr_nodes)

    d = {'level': level, 'node': nodes, 'channel1': channel1, 'channel2': channel2, 'weight1': weight1,
         'weight2': weight2, 'weight3': weight3, 'weight4': weight4, 'weight5': weight5, 'weight6': weight6,
         'weight7': weight7, 'weight8': weight8, 'weight9': weight9, 'bias': bias}

    d.update(graph_node_metrics)

    df = pd.DataFrame(data=d)
    print(df)
    return df


def run_epochs(model, args, train_loader, test_loader, txt=''):
    if args.optimizer == 'ADAM':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                               weight_decay=5e-5)
    elif args.optimizer == 'SGD_NO_MOMENTUM':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                              weight_decay=5e-5)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                              weight_decay=5e-5, momentum=0.9)

    criterion = nn.CrossEntropyLoss().to(device)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 2, T_mult=2)

    epoch_list = []
    test_acc_list = []
    train_acc_list = []
    train_loss_list = []
    max_test_acc = 0
    if not os.path.isdir("reporting"):
        os.mkdir("reporting")

    start_time = time.time()
    with open("./reporting/" + "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_" + args.graph_mode + "_dataset_" +
              args.dataset_mode + "_seed_" + str(args.seed) + "_name_" + args.name + "_opt_" + args.optimizer + txt +
              ".csv", "w") as f:
        f.write('epoch,accuracy,train_loss,train_acc,time\n')
        for epoch in range(1, args.epochs + 1):

            epoch_list.append(epoch)
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch, args)
            test_acc = get_test(model, test_loader)
            test_acc_list.append(test_acc)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            print('Test set accuracy: {0:.2f}%, Best accuracy: {1:.2f}%'.format(test_acc, max_test_acc))
            f.write("{0:3d}, {1:.3f}, {2:.3f}, {3:.3f}".format(epoch, test_acc, train_loss, train_acc))

            if max_test_acc < test_acc:
                print('Saving..')
                state = {
                    'model': model.state_dict(),
                    'acc': test_acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                filename = "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_" + args.graph_mode + "_dataset_" + \
                           args.dataset_mode + "_seed_" + str(args.seed) + "_name_" + args.name
                torch.save(state, './checkpoint/' + filename + txt + '_end_ckpt.t7')
                max_test_acc = test_acc
                draw_plot(epoch_list, train_loss_list, train_acc_list, test_acc_list)
            print("Training time: ", time.time() - start_time)
            f.write("," + str(time.time() - start_time))
            f.write("\n")
