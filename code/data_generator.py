import pickle
import random

import networkx as nx
import numpy as np
import torch
import ipdb

np.random.seed(1)

class GNNGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())

        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])

        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert (type(edge_features.values()[0]) == np.ndarray)
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in edge_features.items()}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)


class DataGenerator(object):
    def __init__(self, args, train_graph_list, test_graph_list):
        self.data_path = args.data_path
        self.batch_n = args.batch_n
        self.task_type = args.task_type
        self.train_graph_list = train_graph_list
        self.test_graph_list = test_graph_list
        self.shot_num = args.shot_num
        self.num_class = args.num_class
        self.train_graphs = self.num_class * self.shot_num
        self.test_graphs = args.test_graphs
        self.edge_feat_dim = args.edge_feat_dim
        self.fea_size = args.fea_size
        self.attr_dim = args.attr_dim

    def load_data(self):

        print('loading data')
        with open('{0}/{1}/element_dict'.format(self.data_path, self.task_type), 'rb') as f:
            feat_dict = pickle.load(f)

        self.meta_train_graphset = {}
        self.meta_test_graphset = {}

        all_graph_list = self.train_graph_list + self.test_graph_list

        for graphset in all_graph_list:
            g_list = []
            label_dict = {}
            with open('{0}/{1}/{1}_{2}.txt'.format(self.data_path, self.task_type, graphset), 'r') as f:
                n_g = int(f.readline().strip())
                for i in range(n_g):
                    row = f.readline().strip().split()
                    n, l = [int(w) for w in row]
                    if not l in label_dict:
                        mapped = len(label_dict)
                        label_dict[l] = mapped
                    g = nx.Graph()
                    node_tags = []
                    node_features = []
                    n_edges = 0
                    for j in range(n):
                        g.add_node(j)
                        row = f.readline().strip().split()
                        tmp = int(row[1]) + 2
                        if tmp == len(row):
                            # no node attributes
                            row = [int(w) for w in row]
                            attr = None
                        else:
                            row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                        node_tags.append(row[0])

                        if attr is not None:
                            node_features.append(attr)

                        n_edges += row[1]
                        for k in range(2, len(row)):
                            g.add_edge(j, row[k])

                    if node_features != []:
                        node_features = np.stack(node_features)
                        node_feature_flag = True
                    else:
                        node_features = None
                        node_feature_flag = False

                    # assert len(g.edges()) * 2 == n_edges  (some graphs in COLLAB have self-loops, ignored here)
                    assert len(g) == n
                    g_list.append(GNNGraph(g, l, node_tags, node_features))
            for g in g_list:
                g.label = label_dict[g.label]
            self.num_class = len(label_dict)

            self.fea_size = len(feat_dict)  # maximum node label (tag)
            if node_feature_flag == True:
                self.attr_dim = node_features.shape[1]  # dim of node features (attributes)
            else:
                self.attr_dim = 0

            # print('# classes: %d' % self.num_class)
            # print('# maximum node tag: %d' % self.fea_size)
            # print('# attribute dim: %d' % self.attr_dim)

            random.shuffle(g_list)

            if graphset in self.train_graph_list:
                self.meta_train_graphset[graphset] = {}
                self.meta_train_graphset[graphset]['pos'] = [x for x in g_list if x.label == 1]
                self.meta_train_graphset[graphset]['neg'] = [x for x in g_list if x.label == 0]
            elif graphset in self.test_graph_list:
                self.meta_test_graphset[graphset] = {}
                self.meta_test_graphset[graphset]['pos'] = [x for x in g_list if x.label == 1]
                self.meta_test_graphset[graphset]['neg'] = [x for x in g_list if x.label == 0]

    def generate_batch(self):
        train_graphs, test_graphs = [], []
        for _ in range(self.batch_n):
            sel_data = self.meta_train_graphset[random.choice([*self.meta_train_graphset.keys()])]
            random.shuffle(sel_data['pos'])
            random.shuffle(sel_data['neg'])
            temp_train=sel_data['pos'][:self.shot_num] + sel_data['neg'][:self.shot_num]
            temp_test=sel_data['pos'][self.shot_num:self.shot_num + self.test_graphs] + sel_data['neg'][
                                                                                            self.shot_num:self.shot_num + self.test_graphs]
            random.shuffle(temp_train)
            random.shuffle(temp_test)
            train_graphs.append(temp_train)
            test_graphs.append(temp_test)
        return train_graphs, test_graphs

    # def generate_batch_all(self):
    #     train_graphs, test_graphs = [], []
    #     train_data = [*self.meta_train_graphset.values()]
    #     for file in train_data:
    #         temp_train = file['pos'] + file['neg']
    #         train_graphs = train_graphs + temp_train
    #     test_data = [*self.meta_test_graphset.values()]
    #     for file in test_data:
    #         temp_test = file['pos'] + file['neg']
    #         test_graphs = test_graphs + temp_test
    #     train_graphs = train_graphs + temp_test
    #     test_graphs = random.sample(test_graphs,self.test_graphs)
    #     return train_graphs, test_graphs
    def generate_batch_all(self):
        meta_graphs = {**self.meta_test_graphset, **self.meta_train_graphset}
        keys = [*meta_graphs.keys()] #keys=[1,33,41,47,81,83,109,123,145]
        train_keys = keys[:6]  #train_keys=[1,33,41,47,81,83]
        test_keys = list(set(keys)-set(train_keys))
        train_graphs, test_graphs = [], []
        for key, value in meta_graphs.items():
            if key in train_keys:
                temp_train = value['pos'] + value['neg']
                train_graphs = train_graphs + temp_train
            elif key in test_keys:
                temp_test = value['pos'] + value['neg']
                test_graphs = test_graphs + temp_test
        return train_graphs, test_graphs



    def generate_test_batch(self):
        train_graphs, test_graphs=[], []
        sel_data = self.meta_test_graphset[109] #test_graph_list=[109,123,145]
        #sel_data = self.meta_test_graphset[random.choice([*self.meta_test_graphset.keys()])]
        random.shuffle(sel_data['pos'])
        random.shuffle(sel_data['neg'])
        temp_train=sel_data['pos'][:self.shot_num] + sel_data['neg'][:self.shot_num]
        temp_test=sel_data['pos'][self.shot_num:self.shot_num + self.test_graphs] + sel_data['neg'][
                                                                                        self.shot_num:self.shot_num + self.test_graphs]
        random.shuffle(temp_train)
        random.shuffle(temp_test)
        train_graphs.append(temp_train)
        test_graphs.append(temp_test)
        return train_graphs, test_graphs

    def PrepareFeatureLabel(self, batch_graph, sel_label):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        if self.edge_feat_dim > 0:
            edge_feat_flag = True
            concat_edge_feat = []
        else:
            edge_feat_flag = False

        for i in range(len(batch_graph)):
            if batch_graph[i].label != sel_label and sel_label != -1:
                continue
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag == True:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag == True:
                tmp = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
                concat_feat.append(tmp)
            if edge_feat_flag == True:
                if batch_graph[i].edge_features is not None:  # in case no edge in graph[i]
                    tmp = torch.from_numpy(batch_graph[i].edge_features).type('torch.FloatTensor')
                    concat_edge_feat.append(tmp)

        if node_tag_flag == True:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, self.fea_size)
            node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag == True:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features

        if edge_feat_flag == True:
            edge_feat = torch.cat(concat_edge_feat, 0)

        node_feat = node_feat.cuda()

        labels = labels.cuda()
        if edge_feat_flag == True:
            edge_feat = edge_feat.cuda()

        if edge_feat_flag == True:
            return node_feat, edge_feat, labels
        return node_feat, labels