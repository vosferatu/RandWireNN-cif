import torch
import argparse
from tqdm import tqdm
from model import Model

import pandas as pd
import randwire
import graph

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
        if step % 100 == 0:
            print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epoch, loss.data, acc), end='')
            for param_group in optimizer.param_groups:
                print(",  Current learning rate is: {}".format(param_group['lr']))

    length = len(train_loader.dataset) // args.batch_size
    return train_loss / length, train_acc / length


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


def get_args():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--epochs', type=int, default=100, help='number of epochs, (default: 100)')
    parser.add_argument('--p', type=float, default=0.75, help='graph probability, (default: 0.75)')
    parser.add_argument('--c', type=int, default=78,
                        help='channel count for each node, (example: 78, 109, 154), (default: 154)')
    parser.add_argument('--k', type=int, default=4,
                        help='each node is connected to k nearest neighbors in ring topology, (default: 4)')
    parser.add_argument('--m', type=int, default=5,
                        help='number of edges to attach from a new node to existing nodes, (default: 5)')
    parser.add_argument('--graph-mode', type=str, default="WS",
                        help="random graph, (Example: ER, WS, BA), (default: WS)")
    parser.add_argument('--node-num', type=int, default=32,
                        help="Number of graph node (default n=32)")
    parser.add_argument('--learning-rate', type=float, default=1e-1,
                        help='learning rate, (default: 1e-1)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size, (default: 100)')
    parser.add_argument('--model-mode', type=str, default="SMALL_REGIME",
                        help='CIFAR10, CIFAR100, SMALL_REGIME, REGULAR_REGIME, (default: CIFAR10)')
    parser.add_argument('--dataset-mode', type=str, default="CIFAR10",
                        help='Which dataset to use? (Example, CIFAR10, CIFAR100, MNIST), (default: CIFAR10)')
    parser.add_argument('--is-train', type=bool, default=False,
                        help="True if training, False if test. (default: False)")
    parser.add_argument('--name', type=str, default='', help='name of the current architecture')
    parser.add_argument('--seed', type=int, default=-1, help='seed in the random graph algorithm')

    args = parser.parse_args()

    return args


def load_model(args):
    model = Model(args.node_num, args.p, args.c, args.c, args.graph_mode, args.model_mode, args.dataset_mode,
                  args.is_train, args.k, args.m, args.name, args.seed).to(device)
    filename = "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_" + args.graph_mode + "_dataset_" + \
               args.dataset_mode + "_seed_" + str(args.seed) + "_name_" + args.name

    if device is 'cuda':
        model = torch.nn.DataParallel(model)

    model.to(device)

    checkpoint = torch.load('./checkpoint/' + filename + 'ckpt.t7')
    model.load_state_dict(checkpoint['model'], strict=True)
    epoch = checkpoint['epoch']
    acc = checkpoint['acc']
    print("Load Model Accuracy: ", acc, "Load Model end epoch: ", epoch)

    return model, filename


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


def main():
    args = get_args()
    # train_loader, test_loader = load_data(args)
    model, filename = load_model(args)

    df = get_dataset(model)

    df.to_csv('./node_data/' + filename + '.csv')


if __name__ == '__main__':
    main()
