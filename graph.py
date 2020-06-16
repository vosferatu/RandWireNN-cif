import os

import networkx as nx


class RandomGraph(object):
    def __init__(self, node_num, p, k=4, m=5, is_train=True, name='', graph_mode="WS", seed=-1):
        self.node_num = node_num
        self.p = p
        self.k = k
        self.m = m
        self.graph_mode = graph_mode
        self.seed = seed
        self.name = name

        if is_train is True:
            print("is_train: True")
            self.graph = self.make_graph()
            self.save_random_graph(self.name)
        else:
            self.graph = self.load_random_graph(self.name)

        self.nodes, self.in_edges = self.get_graph_info()

    def make_graph(self):

        if self.graph_mode == "ER":
            if self.seed == -1:
                my_graph = nx.random_graphs.erdos_renyi_graph(self.node_num, self.p)
            else:
                my_graph = nx.random_graphs.erdos_renyi_graph(self.node_num, self.p, self.seed)

        elif self.graph_mode == "WS":
            if self.seed == -1:
                my_graph = nx.random_graphs.connected_watts_strogatz_graph(self.node_num, self.k, self.p, tries=1000)
            else:
                my_graph = nx.random_graphs.connected_watts_strogatz_graph(self.node_num, self.k, self.p, 1000,
                                                                           self.seed)

        elif self.graph_mode == "BA":
            if self.seed == -1:
                my_graph = nx.random_graphs.barabasi_albert_graph(self.node_num, self.m)
            else:
                my_graph = nx.random_graphs.barabasi_albert_graph(self.node_num, self.m, self.seed)

        return my_graph

    def get_graph_info(self):
        in_edges = {0: []}
        nodes = [0]
        end = []
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            neighbors.sort()

            edges = []
            check = []
            for neighbor in neighbors:
                if node > neighbor:
                    edges.append(neighbor + 1)
                    check.append(neighbor)
            if not edges:
                edges.append(0)
            in_edges[node + 1] = edges
            if check == neighbors:
                end.append(node + 1)
            nodes.append(node + 1)
        in_edges[self.node_num + 1] = end
        nodes.append(self.node_num + 1)

        return nodes, in_edges

    def get_network_graph(self):
        net_graph = nx.Graph()

        for x in self.nodes:
            net_graph.add_node(x)

        for i in self.in_edges:
            for j in self.in_edges[i]:
                net_graph.add_edge(i, j, name=str(i) + '-' + str(j))
        return net_graph

    def get_network_di_graph(self):
        net_graph = nx.DiGraph()

        for x in self.nodes:
            net_graph.add_node(x)

        for i in self.in_edges:
            for j in self.in_edges[i]:
                net_graph.add_edge(i, j, name=str(i) + '-' + str(j))
        return net_graph

    def get_in_degree(self):
        in_degree = []
        for x in self.in_edges:
            in_degree.append(len(self.in_edges[x]))
        return in_degree

    def get_out_degree(self):
        out_degree = []
        for x in range(len(self.in_edges)):
            count = 0
            for a in self.in_edges:
                if x in self.in_edges[a]:
                    count = count + 1
            out_degree.append(count)

        return out_degree

    def get_edge_metrics(self, df, connection):
        net_graph = self.get_network_graph()
        di_graph = self.get_network_di_graph()

        # UNDIRECTED GRAPH
        deg_centrality = nx.degree_centrality(net_graph)
        close_centrality = nx.closeness_centrality(net_graph)
        bet_centrality = nx.betweenness_centrality(net_graph)
        curr_flow_close_centrality = nx.current_flow_closeness_centrality(net_graph)
        curr_flow_bet_centrality = nx.current_flow_betweenness_centrality(net_graph)
        eigen_centrality = nx.eigenvector_centrality(net_graph)
        katz_centrality = nx.katz_centrality(net_graph)
        comm_bet_centrality = nx.communicability_betweenness_centrality(net_graph)
        load_centrality = nx.load_centrality(net_graph)
        page_rank = nx.pagerank(net_graph)
        communicability = nx.communicability(net_graph)
        average_neighbor_degree = nx.average_neighbor_degree(net_graph)
        edge_curr_flow_bet_centrality = nx.edge_current_flow_betweenness_centrality(net_graph)
        edge_load_centrality = nx.edge_load_centrality(net_graph)

        deg_c_a = []
        deg_c_b = []
        close_c_a = []
        close_c_b = []
        bet_c_a = []
        bet_c_b = []
        curr_flow_c_a = []
        curr_flow_c_b = []
        curr_flow_bet_centrality_a = []
        curr_flow_bet_centrality_b = []
        eigen_centrality_a = []
        eigen_centrality_b = []
        katz_centrality_a = []
        katz_centrality_b = []
        comm_bet_centrality_a = []
        comm_bet_centrality_b = []
        load_centrality_a = []
        load_centrality_b = []
        page_rank_a = []
        page_rank_b = []
        dispersion = []
        comm = []
        node_connectivity = []
        edge_connectivity = []
        avg_neighbor_degree_a = []
        avg_neighbor_degree_b = []
        edge_curr_flow_bet_cent = []
        group_bet_cent = []
        group_clo_cent = []
        group_deg_cent = []
        edge_load_cent = []
        simrank_similarity = []
        volume = []
        depth_a = []
        depth_b = []

        # DIRECTED GRAPH
        in_deg_centrality = nx.in_degree_centrality(di_graph)
        out_deg_centrality = nx.out_degree_centrality(di_graph)
        edge_bet_centrality = nx.edge_betweenness_centrality(di_graph)

        in_deg_c_a = []
        in_deg_c_b = []
        out_deg_c_a = []
        out_deg_c_b = []
        edge_bet_cent = []
        group_in_deg_cent = []
        group_out_deg_cent = []

        for i in connection:
            nodes = i.split('-')
            node_a = int(nodes[0])
            node_b = int(nodes[1])

            deg_c_a.append(deg_centrality[node_a])
            deg_c_b.append(deg_centrality[node_b])
            close_c_a.append(close_centrality[node_a])
            close_c_b.append(close_centrality[node_b])
            bet_c_a.append(bet_centrality[node_a])
            bet_c_b.append(bet_centrality[node_b])
            curr_flow_c_a.append(curr_flow_close_centrality[node_a])
            curr_flow_c_b.append(curr_flow_close_centrality[node_b])
            curr_flow_bet_centrality_a.append(curr_flow_bet_centrality[node_a])
            curr_flow_bet_centrality_b.append(curr_flow_bet_centrality[node_b])
            eigen_centrality_a.append(eigen_centrality[node_a])
            eigen_centrality_b.append(eigen_centrality[node_b])
            katz_centrality_a.append(katz_centrality[node_a])
            katz_centrality_b.append(katz_centrality[node_b])
            comm_bet_centrality_a.append(comm_bet_centrality[node_a])
            comm_bet_centrality_b.append(comm_bet_centrality[node_b])
            load_centrality_a.append(load_centrality[node_a])
            load_centrality_b.append(load_centrality[node_b])
            page_rank_a.append(page_rank[node_a])
            page_rank_b.append(page_rank[node_b])
            dispersion.append(nx.dispersion(net_graph, node_a, node_b))
            comm.append(communicability[node_a][node_b])
            node_connectivity.append((nx.node_connectivity(net_graph, node_a, node_b)))
            edge_connectivity.append(nx.edge_connectivity(net_graph, node_a, node_b))
            avg_neighbor_degree_a.append(average_neighbor_degree[node_a])
            avg_neighbor_degree_b.append(average_neighbor_degree[node_b])
            in_deg_c_a.append(in_deg_centrality[node_a])
            in_deg_c_b.append(in_deg_centrality[node_b])
            out_deg_c_a.append(out_deg_centrality[node_a])
            out_deg_c_b.append(out_deg_centrality[node_b])
            edge_bet_cent.append(edge_bet_centrality[node_b, node_a])
            if (node_b, node_a) in edge_curr_flow_bet_centrality.keys():
                edge_curr_flow_bet_cent.append(edge_curr_flow_bet_centrality[node_b, node_a])
            else:
                edge_curr_flow_bet_cent.append(edge_curr_flow_bet_centrality[node_a, node_b])

            group_bet_cent.append(nx.group_betweenness_centrality(net_graph, [node_a, node_b]))
            group_clo_cent.append(nx.group_closeness_centrality(net_graph, [node_a, node_b]))
            group_deg_cent.append(nx.group_degree_centrality(net_graph, [node_a, node_b]))
            group_in_deg_cent.append(nx.group_in_degree_centrality(di_graph, [node_a, node_b]))
            group_out_deg_cent.append(nx.group_out_degree_centrality(di_graph, [node_a, node_b]))
            edge_load_cent.append(edge_load_centrality[node_b, node_a])
            simrank_similarity.append(nx.simrank_similarity(net_graph, node_a, node_b))
            volume.append(nx.volume(net_graph, [node_a, node_b]))
            depth_a.append(nx.shortest_path_length(net_graph, 0, node_a))
            depth_b.append(nx.shortest_path_length(net_graph, 0, node_b))

        df['deg_c_a'] = deg_c_a
        df['deg_c_b'] = deg_c_b
        df['close_c_a'] = close_c_a
        df['close_c_b'] = close_c_b
        df['bet_c_a'] = bet_c_a
        df['bet_c_b'] = bet_c_b
        df['curr_flow_c_a'] = curr_flow_c_a
        df['curr_flow_c_b'] = curr_flow_c_b
        df['curr_flow_bet_centrality_a'] = curr_flow_bet_centrality_a
        df['curr_flow_bet_centrality_b'] = curr_flow_bet_centrality_b
        df['eigen_centrality_a'] = eigen_centrality_a
        df['eigen_centrality_b'] = eigen_centrality_b
        df['katz_centrality_a'] = katz_centrality_a
        df['katz_centrality_b'] = katz_centrality_b
        df['comm_bet_centrality_a'] = comm_bet_centrality_a
        df['comm_bet_centrality_b'] = comm_bet_centrality_b
        df['load_centrality_a'] = load_centrality_a
        df['load_centrality_b'] = load_centrality_b
        df['page_rank_a'] = page_rank_a
        df['page_rank_b'] = page_rank_b
        df['dispersion'] = dispersion
        df['comm'] = comm
        df['connectivity'] = node_connectivity
        df['edge_connectivity'] = edge_connectivity
        df['avg_neighbor_degree_a'] = avg_neighbor_degree_a
        df['avg_neighbor_degree_b'] = avg_neighbor_degree_b
        df['in_deg_c_a'] = in_deg_c_a
        df['in_deg_c_b'] = in_deg_c_b
        df['out_deg_c_a'] = out_deg_c_a
        df['out_deg_c_b'] = out_deg_c_b
        df['edge_bet_cent'] = edge_bet_cent
        df['edge_curr_flow_bet_centrality'] = edge_curr_flow_bet_cent
        df['group_bet_cent'] = group_bet_cent
        df['group_clo_cent'] = group_clo_cent
        df['group_deg_cent'] = group_deg_cent
        df['group_in_deg_cent'] = group_in_deg_cent
        df['group_out_deg_cent'] = group_out_deg_cent
        df['edge_load_cent'] = edge_load_cent
        df['simrank_similarity'] = simrank_similarity
        df['volume'] = volume
        df['depth_a'] = depth_a
        df['depth_b'] = depth_b

        return df

    def get_node_metrics(self, df, nodes):
        net_graph = self.get_network_graph()
        di_graph = self.get_network_di_graph()

        # UNDIRECTED GRAPH
        deg_centrality = nx.degree_centrality(net_graph)
        close_centrality = nx.closeness_centrality(net_graph)
        bet_centrality = nx.betweenness_centrality(net_graph)
        curr_flow_close_centrality = nx.current_flow_closeness_centrality(net_graph)
        curr_flow_bet_centrality = nx.current_flow_betweenness_centrality(net_graph)
        eigen_centrality = nx.eigenvector_centrality(net_graph)
        katz_centrality = nx.katz_centrality(net_graph)
        comm_bet_centrality = nx.communicability_betweenness_centrality(net_graph)
        load_centrality = nx.load_centrality(net_graph)
        page_r = nx.pagerank(net_graph)
        communicability = nx.communicability(net_graph)
        average_neighbor_degree = nx.average_neighbor_degree(net_graph)

        deg_cent = []
        close_cent = []
        bet_cent = []
        curr_flow_cent = []
        curr_flow_bet_cent = []
        eigen_cent = []
        katz_cent = []
        comm_bet_cent = []
        load_cent = []
        page_rank = []
        dispersion = []
        comm = []
        node_connectivity = []
        avg_neighbor_degree_a = []
        group_bet_cent = []
        group_clo_cent = []
        group_deg_cent = []
        volume = []
        depth = []

        # DIRECTED GRAPH
        in_deg_centrality = nx.in_degree_centrality(di_graph)
        out_deg_centrality = nx.out_degree_centrality(di_graph)

        in_deg_cent = []
        out_deg_cent = []
        group_in_deg_cent = []
        group_out_deg_cent = []

        for node in nodes:
            deg_cent.append(deg_centrality[node])
            close_cent.append(close_centrality[node])
            bet_cent.append(bet_centrality[node])
            curr_flow_cent.append(curr_flow_close_centrality[node])
            curr_flow_bet_cent.append(curr_flow_bet_centrality[node])
            eigen_cent.append(eigen_centrality[node])
            katz_cent.append(katz_centrality[node])
            comm_bet_cent.append(comm_bet_centrality[node])
            load_cent.append(load_centrality[node])
            page_rank.append(page_r[node])
            dispersion.append(nx.dispersion(net_graph, 0, node))
            comm.append(communicability[0][node])
            node_connectivity.append((nx.node_connectivity(net_graph, 0, node)))
            avg_neighbor_degree_a.append(average_neighbor_degree[node])
            in_deg_cent.append(in_deg_centrality[node])
            out_deg_cent.append(out_deg_centrality[node])
            group_bet_cent.append(nx.group_betweenness_centrality(net_graph, [0, node]))
            group_clo_cent.append(nx.group_closeness_centrality(net_graph, [0, node]))
            group_deg_cent.append(nx.group_degree_centrality(net_graph, [0, node]))
            group_in_deg_cent.append(nx.group_in_degree_centrality(di_graph, [0, node]))
            group_out_deg_cent.append(nx.group_out_degree_centrality(di_graph, [0, node]))
            volume.append(nx.volume(net_graph, [0, node]))
            depth.append(nx.shortest_path_length(net_graph, 0, node))

        df['deg_cent'] = deg_cent
        df['close_cent'] = close_cent
        df['bet_cent'] = bet_cent
        df['curr_flow_cent'] = curr_flow_cent
        df['curr_flow_bet_cent'] = curr_flow_bet_cent
        df['eigen_cent'] = eigen_cent
        df['katz_cent'] = katz_cent
        df['comm_bet_cent'] = comm_bet_cent
        df['load_cent'] = load_cent
        df['page_rank'] = page_rank
        df['dispersion'] = dispersion
        df['comm'] = comm
        df['connectivity'] = node_connectivity
        df['avg_neighbor_degree_a'] = avg_neighbor_degree_a
        df['in_deg_cent'] = in_deg_cent
        df['out_deg_cent'] = out_deg_cent
        df['group_bet_cent'] = group_bet_cent
        df['group_clo_cent'] = group_clo_cent
        df['group_deg_cent'] = group_deg_cent
        df['group_in_deg_cent'] = group_in_deg_cent
        df['group_out_deg_cent'] = group_out_deg_cent
        df['volume'] = volume
        df['depth'] = depth
        return df

    def save_random_graph(self, path):
        if not os.path.isdir("saved_graph"):
            os.mkdir("saved_graph")
        nx.write_yaml(self.graph, "./saved_graph/" + path)

    def load_random_graph(self, path):
        print(path)
        self.graph = nx.read_yaml("./saved_graph/" + path)
        return self.graph
