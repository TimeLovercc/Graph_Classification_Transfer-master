import os
import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

sys.path.append('%s/lib' % os.path.dirname(os.path.realpath(__file__)))
from utils import process_sparse

from layers import GraphAttentionLayer, GraphConvolution, MatchingLayer
import ipdb


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

    def functional_forward(self, x, adj, weights):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att.functional_forward(x, adj, att_idx, weights) for att_idx, att in enumerate(self.attentions)],
                      dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.nhid = nhid
        self.dropout = dropout

    def forward(self, graph_list, node_feat, edge_feat, eval=False):

        graph_sizes, n2n_sp, e2n_sp, subg_sp, node_degs = process_sparse(graph_list, node_feat, edge_feat)

        x = F.relu(self.gc1(node_feat, n2n_sp, node_degs), inplace=True)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, n2n_sp, node_degs)
        batch_graphs_repr = torch.zeros(len(graph_sizes), self.nhid).cuda()
        batch_graphs_repr = Variable(batch_graphs_repr)
        accum_count = 0
        if eval:
            for i in range(subg_sp.size()[0]):
                batch_graphs_repr[i, :] = torch.mean(x[accum_count: accum_count + graph_sizes[i]], dim=0, keepdim=True)
                accum_count += graph_sizes[i]
            return batch_graphs_repr
        else:
            x = F.dropout(x, self.dropout, training=self.training)
            for i in range(subg_sp.size()[0]):
                batch_graphs_repr[i, :] = torch.mean(x[accum_count: accum_count + graph_sizes[i]], dim=0, keepdim=True)
                accum_count += graph_sizes[i]
            batch_graphs_repr = self.mlp(batch_graphs_repr)
            return F.log_softmax(batch_graphs_repr, dim=1)

    def functional_forward(self, graph_list, node_feat, edge_feat, weights, eval=False):

        graph_sizes, n2n_sp, e2n_sp, subg_sp, node_degs = process_sparse(graph_list, node_feat, edge_feat)

        x = F.relu(self.gc1.functional_forward(node_feat, n2n_sp, node_degs, id=1, weights=weights), inplace=True)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2.functional_forward(x, n2n_sp, node_degs, id=2, weights=weights)
        batch_graphs_repr = torch.zeros(len(graph_sizes), self.nhid).cuda()
        batch_graphs_repr = Variable(batch_graphs_repr)
        accum_count = 0
        if eval:
            for i in range(subg_sp.size()[0]):
                batch_graphs_repr[i, :] = torch.mean(x[accum_count: accum_count + graph_sizes[i]], dim=0, keepdim=True)
                accum_count += graph_sizes[i]
            return batch_graphs_repr
        else:
            x = F.dropout(x, self.dropout, training=self.training)
            for i in range(subg_sp.size()[0]):
                batch_graphs_repr[i, :] = torch.mean(x[accum_count: accum_count + graph_sizes[i]], dim=0, keepdim=True)
                accum_count += graph_sizes[i]
            batch_graphs_repr = torch.mm(batch_graphs_repr, weights['mlp.weight'].t()) + weights['mlp.bias']
            return F.log_softmax(batch_graphs_repr, dim=1)


class GraphMatching(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout):
        super(GraphMatching, self).__init__()

        self.nhid = nhid
        self.args = args
        self.dropout = dropout
        self.nclass = nclass
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.message = GraphConvolution(nhid * 2, nhid)
        self.aggregate = nn.Linear(nhid * 2, nhid)
        self.match = MatchingLayer(nhid, nhid)
        self.final = nn.ModuleList([nn.Linear(nhid, 1), nn.Linear(nhid, 1)])

    def forward(self, query, support, query_node_feat, support_node_feat, query_edge_feat, support_edge_feat,
                weights, id, eval=False):
        query_graph_sizes, query_n2n_sp, query_e2n_sp, query_subg_sp, query_node_degs = process_sparse(query,
                                                                                                       query_node_feat,
                                                                                                       query_edge_feat)
        support_graph_sizes, support_n2n_sp, support_e2n_sp, support_subg_sp, support_node_degs = process_sparse(
            support, support_node_feat, support_edge_feat)

        support_x = F.relu(self.gc1(support_node_feat, support_n2n_sp, support_node_degs), inplace=True)
        support_x = F.dropout(support_x, self.dropout, training=self.training)

        query_x = F.relu(self.gc1(query_node_feat, query_n2n_sp, query_node_degs), inplace=True)
        query_x = F.dropout(query_x, self.dropout, training=self.training)

        res_query_match = []

        query_accum_count = 0

        for query_graph_id in range(query_subg_sp.size()[0]):
            query_cat = query_x[query_accum_count: query_accum_count + query_graph_sizes[query_graph_id]]
            support_vec = support_x
            for layer_id in range(self.args.layer):

                support_accum_count = 0
                layer_matching_vec = []
                support_vec_temp = []
                for support_graph_id in range(support_subg_sp.size()[0]):
                    # if support_graph_id==0:
                    #     ipdb.set_trace()
                    matching_vec = self.match(
                        query_cat,
                        support_vec[support_accum_count: support_accum_count + support_graph_sizes[support_graph_id]])
                    matching_vec = torch.sum(matching_vec, dim=1)
                    layer_matching_vec.append(matching_vec)
                    matching_vec_support = self.match(support_vec[
                                                      support_accum_count: support_accum_count + support_graph_sizes[
                                                          support_graph_id]], query_cat)
                    matching_vec_support = torch.sum(matching_vec_support, dim=1)
                    support_cat = torch.cat((matching_vec_support, support_vec[
                                                                   support_accum_count: support_accum_count +
                                                                                        support_graph_sizes[
                                                                                            support_graph_id]]),
                                            dim=-1)
                    support_vec_temp.append(support_cat)
                    support_accum_count = support_accum_count + support_graph_sizes[support_graph_id]
                support_cat = torch.cat(support_vec_temp, dim=0)
                support_vec = self.aggregate(support_cat)
                layer_matching_vec = F.relu(torch.mean(torch.stack(layer_matching_vec, dim=0), dim=0, keepdim=False),
                                            inplace=True)
                query_cat = torch.cat((layer_matching_vec, query_cat), dim=-1)
                query_cat = self.aggregate(query_cat)
            # ipdb.set_trace()
            res_query_match.append(torch.mean(self.final[id](query_cat), dim=0, keepdim=True))
            query_accum_count = query_accum_count + query_graph_sizes[query_graph_id]
        res_query_match = torch.cat(res_query_match)

        return res_query_match
