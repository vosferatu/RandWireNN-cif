import argparse
import os

import torch
from tqdm import tqdm

import train_utils
from model import Model
from preprocess import load_data

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def main():
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument('--p', type=float, default=0.75, help='graph probability, (default: 0.4)')
    parser.add_argument('--c', type=int, default=78,
                        help='channel count for each node, (example: 78, 109, 154), (default: 154)')
    parser.add_argument('--k', type=int, default=4,
                        help='each node is connected to k nearest neighbors in ring topology, (Default: 4)')
    parser.add_argument('--m', type=int, default=5,
                        help='number of edges to attach from a new node to existing nodes, (default: 5)')
    parser.add_argument('--graph-mode', type=str, default="WS",
                        help="random graph, (exampple: ER, WS, BA), (default: WS)")
    parser.add_argument('--node-num', type=int, default=32,
                        help="number of graph node (default n=32)")
    parser.add_argument('--model-mode', type=str, default="CIFAR10",
                        help='which network you use, (example: CIFAR10, SMALL_REGIME, etc), (default: CIFAR10)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='batch size, (default: 100)')
    parser.add_argument('--dataset-mode', type=str, default="CIFAR10",
                        help="which dataset you use, (example: CIFAR10, CIFAR100, MNIST), (default: CIFAR10)")
    parser.add_argument('--is-train', type=bool, default=False,
                        help="True if training, False if test. (default: False)")
    parser.add_argument('--name', type=str, default='', help='name of the current architecture')
    parser.add_argument('--seed', type=int, default=-1, help='seed in the random graph algorithm')

    args = parser.parse_args()

    _, test_loader = load_data(args)

    if os.path.exists("./checkpoint"):
        model = Model(args.node_num, args.p, args.c, args.c, args.graph_mode, args.model_mode, args.dataset_mode,
                      args.is_train, args.k, args.m, args.name, args.seed).to(device)
        filename = "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_" + args.graph_mode + "_dataset_" + \
                   args.dataset_mode + "_seed_" + str(args.seed) + "_name_" + args.name
        checkpoint = torch.load('./checkpoint/' + filename + 'ckpt.t7')
        model.load_state_dict(checkpoint['model'])
        end_epoch = checkpoint['epoch']
        best_acc = checkpoint['acc']
        print("[Saved Best Accuracy]: ", best_acc, '%', "[End epochs]: ", end_epoch)

        model.eval()
        correct = 0
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            y_pred = output.data.max(1)[1]
            correct += y_pred.eq(target.data).sum()
        print("[Test Accuracy] ", 100. * float(correct) / len(test_loader.dataset), '%')

    else:
        assert os.path.exists("./checkpoint/" + str(args.seed) + "ckpt.t7"), "File not found. Please check again."
    print("Number of model parameters: ", train_utils.get_model_parameters(model))


if __name__ == "__main__":
    main()
