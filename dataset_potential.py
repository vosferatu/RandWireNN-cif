import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import node_data.forest_run as forest
import train_utils
from model import Model
from plot import draw_plot
from preprocess import load_data

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def train(model, train_loader, optimizer, criterion, epoch, args, scheduler):
    model.train()
    step = 0
    train_loss = 0
    train_acc = 0
    for data, target in tqdm(train_loader, desc="epoch " + str(epoch), mininterval=1):
        # train_utils.adjust_learning_rate(optimizer, epoch, args)
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
        scheduler.step(epoch + step / args.batch_size)

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


def run_predicted_eval(model, test_loader):
    df = train_utils.get_dataset(model)
    new_weights = forest.predict_weights(df)
    train_utils.adjust_weights(model, new_weights)

    acc = get_test(model, test_loader)
    print('predicted_weight accuracy: {0:.2f}%'.format(acc))


def run_direct_eval(model, test_loader):
    acc = get_test(model, test_loader)
    print('random_weight accuracy: {0:.2f}%'.format(acc))


def run_epochs(model, args, train_loader, test_loader):
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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, T_mult=2)

    epoch_list = []
    test_acc_list = []
    train_acc_list = []
    train_loss_list = []
    max_test_acc = 0
    if not os.path.isdir("reporting"):
        os.mkdir("reporting")

    start_time = time.time()
    with open("./reporting/" + "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_" + args.graph_mode + "_dataset_" +
              args.dataset_mode + "_seed_" + str(args.seed) + "_name_" + args.name + "_opt_" + args.optimizer +
              str(args.opt) + ".csv", "w") as f:
        f.write('epoch,accuracy,train_loss,train_acc,time\n')
        for epoch in range(1, args.epochs + 1):

            epoch_list.append(epoch)
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch, args, scheduler)
            test_acc = get_test(model, test_loader)
            test_acc_list.append(test_acc)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            print('Test set accuracy: {0:.2f}%, Best accuracy: {1:.2f}%'.format(test_acc, max_test_acc))
            f.write("{0:3d},{1:.3f},{1:.3f},{1:.3f}".format(epoch, test_acc, train_loss, train_acc))

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
                torch.save(state, './checkpoint/' + filename + '_end_ckpt.t7')
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
            'acc': run_direct_eval(model, test_loader)
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        filename = "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_" + args.graph_mode + "_dataset_" + \
                   args.dataset_mode + "_seed_" + str(args.seed) + "_name_" + args.name
        torch.save(state, './checkpoint/' + filename + '_init_ckpt.t7')

    if device is 'cuda':
        model = torch.nn.DataParallel(model)

    if args.opt:
        run_predicted_eval(model, test_loader)
    else:
        run_direct_eval(model, test_loader)

    if args.freeze:
        train_utils.freeze_weights(model)

    train_utils.check_weights(model)

    if args.epochs > 0:
        run_epochs(model, args, train_loader, test_loader)


if __name__ == '__main__':
    main()
