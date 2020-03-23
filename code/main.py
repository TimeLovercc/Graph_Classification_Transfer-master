import argparse
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import sys
from os.path import dirname

sys.path.append('./lib')
import ipdb
import numpy as np
import torch

from data_generator import DataGenerator
from maml import meta_gradient_step, meta_gradient_step_matching, fine_tune
from models import GCN, GraphMatching

parser = argparse.ArgumentParser(description='graph transfer')
parser.add_argument("--data_path", default="../data/", type=str)
parser.add_argument('--method', default='matchingmeta', type=str)
parser.add_argument("--task_type", default="NCI", type=str)
parser.add_argument("--num_class", type=int, default=0, help='#classes')
parser.add_argument("--attr_dim", type=int, default=3)
parser.add_argument("--fea_size", default=4, type=int, help='#max_size_of_fea_node')
parser.add_argument("--test_graphs", default=20, type=int)
parser.add_argument('--logdir', type=str, default='/home/zhimengguo/tmp/', help='directory for summaries and checkpoints.')

# training hyperparameters
parser.add_argument('--batch_n', type=int, default=4, help='meta batch size')
parser.add_argument('--meta_lr', type=float, default=0.01, help='meta learning rate')
parser.add_argument("--shot_num", default=5, type=int)
parser.add_argument('--model', type=str, default='gcn', help='gcn/gat/graphsage')
parser.add_argument('--metatrain_iterations', type=int, default=3000, help='meta training iterations')
parser.add_argument('--metatest_iterations', type=int, default=100, help='meta training iterations')
parser.add_argument('--update_batch_size', type=int, default=100, help='how much samples used for training')
parser.add_argument('--inner_train_steps', type=int, default=5, help='inner_train_step')
parser.add_argument('--inner_test_steps', type=int, default=5, help='inner_test_steps')
parser.add_argument('--inner_lr', type=float, default=1e-3, help='inner learning rate')
parser.add_argument('--inner_test_lr', type=float, default=1e-3, help='inner learning rate of test')
parser.add_argument("--test_load_epoch", default=100, type=int)
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--train', type=int, default=1, help='train or test')
parser.add_argument('--layer', type=int, default=3, help='layers of hierarchical structure')

# model hyperparameters
parser.add_argument('--use_maml', default=0, type=int, help='use maml or not')
parser.add_argument('--use_proto', default=0, type=int, help='use prototype or not')
parser.add_argument("--edge_feat_dim", default=0, type=int)
parser.add_argument('--visualization', type=int, default=0, help='visualization of prototype')
parser.add_argument('--vis_gate', type=int, default=0, help='visualization for gate value')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--structure_dim', type=int, default=8, help='structure dimension')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')

args = parser.parse_args()
print(args)

assert torch.cuda.is_available()
device = torch.device('cuda:{}'.format(args.device))
torch.backends.cudnn.benchmark = True

SAVE_EPOCH = 20

random.seed(0)
np.random.seed(1)

exp_string = "method.{}_".format(args.method) + "metalr.{}_".format(args.meta_lr) + "tasktype.{}_".format(
    args.task_type) + "shotnum.{}_".format(
    args.shot_num) + "proto.{}_".format(args.use_proto) + "maml.{}_".format(args.use_maml) + "batchsize.{}".format(
    args.batch_n)


def train(args, model, optimiser, metatrain_iterations=2000, data_generator=None, verbose=True, fit_function=None,
          fit_function_kwargs={}):
    if verbose:
        print('Begin training...')

    for epoch in range(metatrain_iterations):


        if args.use_maml == 0 and args.use_proto ==0:
            input_data = data_generator.generate_batch_all()
        else:
            input_data = data_generator.generate_batch()

        loss, acc, _ = fit_function(args, data_generator, model, optimiser, input_data,
                                    **fit_function_kwargs)
        print(epoch, loss.item(), acc.item())

        if not os.path.exists(args.logdir + '/' + exp_string + '/'):
            os.makedirs(args.logdir + '/' + exp_string + '/')

        if epoch % SAVE_EPOCH == 0 and epoch != 0:
            torch.save(model.state_dict(), args.logdir + '/' + exp_string + '/' + 'model_epoch_{}'.format(epoch))

    if verbose:
        print('Finished.')


def evaluate(args, model, optimiser, metatest_iterations, data_generator=None,
             verbose=True, fit_function=None, fit_function_kwargs={}):
    if verbose:
        print('Begin evaluating...')

    loss_test, acc_test = [], []

    for epoch in range(metatest_iterations):
        input_data = data_generator.generate_test_batch()
        loss, acc, _ = fit_function(args, data_generator, model, optimiser, input_data, **fit_function_kwargs)
        loss_test.append(loss.item())
        acc_test.append(acc.item())
    loss_test, acc_test = np.array(loss_test), np.array(acc_test)
    # ipdb.set_trace()
    print("testing results: loss is {}, acc is {}, ci is {}".format(np.mean(loss_test), np.mean(acc_test),
                                                                    np.std(acc_test) * 1.96 / np.sqrt(
                                                                        metatest_iterations)))

    if verbose:
        print('Finished.')


def main():
    data_generator = DataGenerator(args, train_graph_list=[1,33,41,47,81,83], test_graph_list=[109,123,145])
    data_generator.load_data()

    if args.method=='matchingmeta':
        meta_model=GraphMatching(args=args, nfeat=data_generator.attr_dim + data_generator.fea_size,
                     nhid=args.hidden,
                     nclass=data_generator.num_class,
                     dropout=args.dropout).to(device)
        fit_function=meta_gradient_step_matching

    elif args.method == 'maml' or args.method == 'proto':
        meta_model = GCN(nfeat=data_generator.attr_dim + data_generator.fea_size,
                         nhid=args.hidden,
                         nclass=data_generator.num_class,
                         dropout=args.dropout).to(device)
        fit_function=meta_gradient_step
    else:
        meta_model = GCN(nfeat=data_generator.attr_dim + data_generator.fea_size,
                         nhid=args.hidden,
                         nclass=data_generator.num_class,
                         dropout=args.dropout).to(device)
        fit_function = fine_tune

    if args.train:
        meta_optimiser = torch.optim.Adam(list(meta_model.parameters()),
                                          lr=args.meta_lr, weight_decay=args.weight_decay)

        train(args, meta_model, meta_optimiser,
              metatrain_iterations=args.metatrain_iterations,
              data_generator=data_generator, fit_function=fit_function,
              fit_function_kwargs={'train': True, 'inner_train_steps': args.inner_train_steps,
                                   'inner_lr': args.inner_lr, 'batch_n': args.batch_n, 'device': device})
    else:
        if args.test_load_epoch > 0:
            meta_model.load_state_dict(
                torch.load(args.logdir + '/' + exp_string + '/' + 'model_epoch_{}'.format(args.test_load_epoch)))

        evaluate(args, meta_model, None, metatest_iterations=args.metatest_iterations,
                 data_generator=data_generator, fit_function=fit_function,
                 fit_function_kwargs={'train': False, 'inner_train_steps': args.inner_test_steps,
                                      'inner_lr': args.inner_test_lr, 'batch_n': 1, 'device': device})


if __name__ == '__main__':
    main()
