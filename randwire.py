import torch
import torch.nn as nn

from graph import RandomGraph


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                              bias=bias)
        self.point_wise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

        # self.apply(weights_init)

    def forward(self, x):
        x = self.conv(x)
        x = self.point_wise(x)
        return x


# ReLU-convolution-BN triplet
class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Unit, self).__init__()

        self.dropout_rate = 0.2

        self.unit = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(in_channels, out_channels, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, x):
        return self.unit(x)


class Node(nn.Module):
    def __init__(self, node, in_edges, in_channels, out_channels, stride=1):
        super(Node, self).__init__()
        self.in_edges = in_edges
        self.node = node
        self.weights = None

        if len(self.in_edges) > 1:
            self.weights = nn.Parameter(torch.ones(len(self.in_edges), requires_grad=True))
        self.unit = Unit(in_channels, out_channels, stride=stride)

    def forward(self, *input):
        # print('self.in_edges), self.in_edges)
        if len(self.in_edges) > 1:
            # print('weights_out: ', self.weights)
            x = (input[0] * torch.sigmoid(self.weights[0]))

            for index in range(1, len(input)):
                x += (input[index] * torch.sigmoid(self.weights[index]))
            out = self.unit(x)

            # different paper, add identity mapping
            # out += x
        else:
            # print('self.unit: ', self.unit)
            out = self.unit(input[0])
        return out


class RandWire(nn.Module):
    def __init__(self, node_num, p, k, m, in_channels, out_channels, graph_mode, is_train, seed, name):
        super(RandWire, self).__init__()
        self.node_num = node_num
        self.p = p
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_mode = graph_mode
        self.is_train = is_train
        if seed != -1:
            self.name = name + '_seed_' + str(seed)
        else:
            self.name = name
        self.seed = seed
        self.k = k
        self.m = m

        # get graph nodes and in edges
        self.graph = RandomGraph(self.node_num, self.p, self.k, self.m, is_train, self.name, self.graph_mode, self.seed)
        self.nodes, self.in_edges = self.graph.get_graph_info()

        # define input Node
        self.module_list = nn.ModuleList([Node(self.nodes[0], self.in_edges[0], self.in_channels, self.out_channels,
                                               stride=2)])
        # define the rest Node
        self.module_list.extend([Node(node, self.in_edges[node], self.out_channels, self.out_channels)
                                 for node in self.nodes if node > 0])

    def forward(self, x):
        # from pudb import set_trace; set_trace()
        memory = {}
        # start vertex
        out = self.module_list[0].forward(x)
        memory[0] = out

        # the rest vertex
        for node in range(1, len(self.nodes) - 1):
            # print(node, self.in_edges[node][0], self.in_edges[node])
            if len(self.in_edges[node]) > 1:
                out = self.module_list[node].forward(*[memory[in_vertex] for in_vertex in self.in_edges[node]])
            else:
                out = self.module_list[node].forward(memory[self.in_edges[node][0]])
            memory[node] = out

        # Reporting 3,
        # How do I handle the last part?
        # It has two kinds of methods.
        # first, Think of the last module as a Node and collect the data by proceeding in the same way
        # as the previous operation.
        # second, simply sum the data and export the output.

        # My Opinion
        # out = self.module_list[self.node_num + 1].forward(*[memory[in_vertex]
        # for in_vertex in self.in_edges[self.node_num + 1]])

        # In paper
        # print("self.in_edges: ", self.in_edges[self.node_num + 1], self.in_edges[self.node_num + 1][0])
        out = memory[self.in_edges[self.node_num + 1][0]]
        for in_vertex_index in range(1, len(self.in_edges[self.node_num + 1])):
            out += memory[self.in_edges[self.node_num + 1][in_vertex_index]]
        out = out / len(self.in_edges[self.node_num + 1])

        # print('outRAND: ', out.shape)

        return out
