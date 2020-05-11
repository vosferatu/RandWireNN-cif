import argparse
import collections

import pandas as pd
import torch

import randwire


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for value in list(layer.size()):
            layer_parameter *= value
        total_parameters += layer_parameter
    return total_parameters


def check_weights(model):
    for graph_conv in model.children():
        if type(graph_conv[0]) == randwire.RandWire:

            for node in graph_conv[0].module_list:
                if len(node.in_edges) > 1:
                    for y in range(len(node.in_edges)):
                        print(str(node.in_edges[y]) + '-' + str(node.node))
                        print(node.weights)


def adjust_weights(model, data, train_weights=True):
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


def get_dataset(model):
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

            graph_conv[0].graph.get_metrics(metrics, curr_edges)

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
