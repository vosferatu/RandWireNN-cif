import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import argparse
import os

import time
from tqdm import tqdm
import pandas as pd
import numpy as np

from model import Model
from preprocess import load_data
from plot import draw_plot
import randwire

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

DATASET = "CIFAR10"


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


def check_weights(model):
    for graph_conv in model.children():
        if type(graph_conv[0]) == randwire.RandWire:

            for node in graph_conv[0].module_list:
                if len(node.in_edges) > 1:
                    for y in range(len(node.in_edges)):
                        print(str(node.in_edges[y]) + '-' + str(node.node))
                        print(node.weights)


def adjust_weights(model, seed):
    data = pd.read_csv(f'./node_data/rand_forest/new_first_gen_{seed}.csv')
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
                    node.weights = torch.nn.Parameter(torch.cuda.FloatTensor(node_weights[node.node]))


def train_opt_sgd_default(args, seed):
    args.seed = seed

    train_loader, test_loader = load_data(args)

    model = Model(args.node_num, args.p, args.c, args.c, args.graph_mode, args.model_mode, args.dataset_mode,
                  args.is_train, args.k, args.m, args.name, args.seed).to(device)
    model.output = nn.Sequential(
            nn.Dropout(model.dropout_rate),
            nn.Linear(1280, 10)
        )
    filename = "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_" + args.graph_mode + "_dataset_" + \
               DATASET + "_seed_" + str(args.seed) + "_name_" + args.name
    checkpoint = torch.load('./checkpoint/' + filename + 'ckpt.t7')
    model.load_state_dict(checkpoint['model'])

    model.output = nn.Sequential(
            nn.Dropout(model.dropout_rate),
            nn.Linear(1280, 100)
        )
    epoch = checkpoint['epoch']
    acc = checkpoint['acc']
    print("Load Model Accuracy: ", acc, "Load Model end epoch: ", epoch)

    adjust_weights(model, seed)

    if device is 'cuda':
        model = torch.nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=5e-5, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    epoch_list = []
    test_acc_list = []
    train_acc_list = []
    train_loss_list = []
    max_test_acc = 0
    if not os.path.isdir("reporting"):
        os.mkdir("reporting")

    start_time = time.time()
    with open("./reporting/" + "new_sgd_default_cifar_100_" + str(args.c) + "_p_" + str(
            args.p) + "_graph_" + args.graph_mode + "_dataset_" +
              args.dataset_mode + "_seed_" + str(args.seed) + "_name_" + args.name + ".csv", "w") as f:
        f.write('epoch,accuracy,time\n')
        for epoch in range(1, args.epochs + 1):
            # scheduler = CosineAnnealingLR(optimizer, epoch)
            epoch_list.append(epoch)
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch, args)
            test_acc = get_test(model, test_loader)
            test_acc_list.append(test_acc)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            print('Test set accuracy: {0:.2f}%, Best accuracy: {1:.2f}%'.format(test_acc, max_test_acc))
            f.write("{0:3d},{1:.3f}".format(epoch, test_acc))

            if max_test_acc < test_acc:
                print('Saving..')
                state = {
                    'model': model.state_dict(),
                    'acc': test_acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                filename = "new_sgd_default_cifar_100_" + str(args.c) + "_p_" + str(
                    args.p) + "_graph_" + args.graph_mode + \
                           "_dataset_" + args.dataset_mode + "_seed_" + str(args.seed) + "_name_" + args.name
                torch.save(state, './checkpoint/' + filename + 'ckpt.t7')
                max_test_acc = test_acc
                draw_plot(epoch_list, train_loss_list, train_acc_list, test_acc_list)
            print("Training time: ", time.time() - start_time)
            f.write("," + str(time.time() - start_time))
            f.write("\n")


def train_opt_sgd_no_momentum(args, seed):
    args.seed = seed

    train_loader, test_loader = load_data(args)

    model = Model(args.node_num, args.p, args.c, args.c, args.graph_mode, args.model_mode, args.dataset_mode,
                  args.is_train, args.k, args.m, args.name, args.seed).to(device)

    model.output = nn.Sequential(
            nn.Dropout(model.dropout_rate),
            nn.Linear(1280, 10)
        )
    filename = "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_" + args.graph_mode + "_dataset_" + \
               DATASET + "_seed_" + str(args.seed) + "_name_" + args.name
    checkpoint = torch.load('./checkpoint/' + filename + 'ckpt.t7')
    model.load_state_dict(checkpoint['model'])
    model.output = nn.Sequential(
            nn.Dropout(model.dropout_rate),
            nn.Linear(1280, 100)
        )
    epoch = checkpoint['epoch']
    acc = checkpoint['acc']
    print("Load Model Accuracy: ", acc, "Load Model end epoch: ", epoch)

    adjust_weights(model, seed)

    if device is 'cuda':
        model = torch.nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=5e-5)
    criterion = nn.CrossEntropyLoss().to(device)

    epoch_list = []
    test_acc_list = []
    train_acc_list = []
    train_loss_list = []
    max_test_acc = 0
    if not os.path.isdir("reporting"):
        os.mkdir("reporting")

    start_time = time.time()
    with open("./reporting/" + "new_sgd_no_momentum_cifar_100_" + str(args.c) + "_p_" + str(
            args.p) + "_graph_" + args.graph_mode + "_dataset_" +
              args.dataset_mode + "_seed_" + str(args.seed) + "_name_" + args.name + ".csv", "w") as f:
        f.write('epoch,accuracy,time\n')
        for epoch in range(1, args.epochs + 1):
            # scheduler = CosineAnnealingLR(optimizer, epoch)
            epoch_list.append(epoch)
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch, args)
            test_acc = get_test(model, test_loader)
            test_acc_list.append(test_acc)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            print('Test set accuracy: {0:.2f}%, Best accuracy: {1:.2f}%'.format(test_acc, max_test_acc))
            f.write("{0:3d},{1:.3f}".format(epoch, test_acc))

            if max_test_acc < test_acc:
                print('Saving..')
                state = {
                    'model': model.state_dict(),
                    'acc': test_acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                filename = "new_sgd_no_momentum_cifar_100_" + str(args.c) + "_p_" + str(
                    args.p) + "_graph_" + args.graph_mode + \
                           "_dataset_" + args.dataset_mode + "_seed_" + str(args.seed) + "_name_" + args.name
                torch.save(state, './checkpoint/' + filename + 'ckpt.t7')
                max_test_acc = test_acc
                draw_plot(epoch_list, train_loss_list, train_acc_list, test_acc_list)
            print("Training time: ", time.time() - start_time)
            f.write("," + str(time.time() - start_time))
            f.write("\n")


def main():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--epochs', type=int, default=50, help='number of epochs, (default: 100)')
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
    parser.add_argument('--dataset-mode', type=str, default="CIFAR100",
                        help='Which dataset to use? (Example, CIFAR10, CIFAR100, MNIST), (default: CIFAR10)')
    parser.add_argument('--is-train', type=bool, default=False,
                        help="True if training, False if test. (default: True)")
    parser.add_argument('--name', type=str, default='first_gen', help='name of the current architecture')
    parser.add_argument('--seed', type=int, default=1, help='seed in the random graph algorithm')

    args = parser.parse_args()

    for network in range(2, 11):
        train_opt_sgd_default(args, network)
        train_opt_sgd_no_momentum(args, network)


if __name__ == '__main__':
    main()