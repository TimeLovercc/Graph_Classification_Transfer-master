import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from utils import process_sparse
import ipdb

import sys
import os

sys.path.append('%s/lib' % os.path.dirname(os.path.realpath(__file__)))
from gnn_lib import GNNLIB
from pytorch_util import weights_init, gnn_spmm


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def functional_forward(self, input, adj, id, weights):
        if id >= 0:
            h = torch.mm(input, weights['attention_{}.W'.format(id)])
        elif id == -1:
            h = torch.mm(input, weights['out_att.W'.format(id)])
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        if id >= 0:
            e = self.leakyrelu(torch.matmul(a_input, weights['attention_{}.a'.format(id)]).squeeze(2))
        elif id == -1:
            e = self.leakyrelu(torch.matmul(a_input, weights['out_att.a'.format(id)]).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, node_degs):
        support = torch.spmm(adj, input) + input
        node_linear = torch.mm(support, self.weight)
        output = node_linear.div(node_degs)
        # support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def functional_forward(self, input, adj, node_degs, id, weights):
        support = torch.spmm(adj, input) + input
        node_linear = torch.mm(support, weights['gc{}.weight'.format(id)]).cuda()
        output = node_linear.div(node_degs)
        # support = torch.mm(input, weights['gc{}.weight'.format(id)])
        # output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + weights['gc{}.bias'.format(id)]
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphPooling(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, node_degs):
        support = torch.spmm(adj, input) + input
        node_linear = torch.mm(support, self.weight)
        output = node_linear.div(node_degs)
        # support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def functional_forward(self, input, adj, node_degs, id, weights):
        support = torch.spmm(adj, input) + input
        node_linear = torch.mm(support, weights['gc{}.weight'.format(id)])
        output = node_linear.div(node_degs)
        # support = torch.mm(input, weights['gc{}.weight'.format(id)])
        # output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + weights['gc{}.bias'.format(id)]
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MatchingLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MatchingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, query_x, support_x):
        dist=torch.mm(query_x, support_x.t())
        att=nn.Softmax(dim=1)(dist)
        query_x=torch.unsqueeze(query_x, dim=1)
        support_x=torch.unsqueeze(support_x, dim=0)
        att = torch.unsqueeze(att, dim=-1)
        matching_value = att*(query_x - support_x)

        return matching_value


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class BidirectionalLSTM(nn.Module):
    def __init__(self, args, layer_size, batch_size, vector_dim):
        super(BidirectionalLSTM, self).__init__()
        self.args = args
        self.batch_size = batch_size
        self.hidden_size = layer_size[0]
        self.vector_dim = vector_dim
        self.num_layer = len(layer_size)
        self.lstm = nn.LSTM(input_size=self.vector_dim, num_layers=self.num_layer, hidden_size=self.hidden_size,
                            bidirectional=True)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (
            Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),
                     requires_grad=False).to(
                self.args.device),
            Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),
                     requires_grad=False).to(
                self.args.device))

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def forward(self, inputs):
        # self.hidden = self.init_hidden(self.use_cuda)
        self.hidden = self.repackage_hidden(self.hidden)
        output, self.hidden = self.lstm(inputs, self.hidden)
        return output
