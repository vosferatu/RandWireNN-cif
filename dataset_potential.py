import argparse
import os

import torch

import train_utils
from model import Model
from preprocess import load_data

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


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
                        help='Which dataset to use? (Example, CIFAR10, CIFAR100, MNIST, IMAGENET), (default: CIFAR10)')
    parser.add_argument('--is-train', type=train_utils.str2bool, nargs='?',
                        const=True, default=True,
                        help="True if training, False if test. (default: True)")
    parser.add_argument('--load-model', type=train_utils.str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--name', type=str, default='first_gen', help='name of the current architecture')
    parser.add_argument('--seed', type=int, default=-1, help='seed in the random graph algorithm')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Type of optimizer: SGD, SGD_NO_MOMENTUM, ADAM')
    parser.add_argument('--opt', type=int, default=0, help='yes is 1, no anything else')
    parser.add_argument('--freeze', type=train_utils.str2bool, default=False, help='yes is 1, no anything else')

    args = parser.parse_args()

    train_loader, test_loader = load_data(args)

    if args.load_model:
        model = Model(args.node_num, args.p, args.c, args.c, args.graph_mode, args.model_mode, args.dataset_mode,
                      args.is_train, args.k, args.m, args.name + '_' + args.dataset_mode, args.seed).to(device)
        filename = "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_" + args.graph_mode + "_dataset_" + \
                   args.dataset_mode + "_seed_" + str(args.seed) + "_name_" + args.name
        print(filename)
        checkpoint = torch.load('./checkpoint/' + filename + '_init_ckpt.t7')
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        acc = checkpoint['acc']
        print("Load Model Accuracy: ", acc, "Load Model end epoch: ", epoch)
    else:
        model = Model(args.node_num, args.p, args.c, args.c, args.graph_mode, args.model_mode, args.dataset_mode,
                      args.is_train, args.k, args.m, args.name + '_' + args.dataset_mode, args.seed).to(device)
        state = {
            'model': model.state_dict(),
            'epoch': 0,
            'acc': train_utils.run_direct_eval(model, test_loader)
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        filename = "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_" + args.graph_mode + "_dataset_" + \
                   args.dataset_mode + "_seed_" + str(args.seed) + "_name_" + args.name
        torch.save(state, './checkpoint/' + filename + '_init_ckpt.t7')

    if device == 'cuda':
        model = torch.nn.DataParallel(model)

    if args.opt:
        train_utils.run_predicted_eval(model, test_loader)
    else:
        train_utils.run_direct_eval(model, test_loader)

    if args.freeze:
        train_utils.freeze_weights(model)

    if args.epochs > 0:
        train_utils.run_epochs(model, args, train_loader, test_loader)


if __name__ == '__main__':
    main()
