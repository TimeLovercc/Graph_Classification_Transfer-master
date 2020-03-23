from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from utils import accuracy
import ipdb


def replace_grad(parameter_gradients, parameter_name):
    def replace_grad_(module):
        return parameter_gradients[parameter_name]

    return replace_grad_


def judge_feat_label(feat_label):
    if len(feat_label) == 2:
        node_feat, labels = feat_label
        edge_feat = None
    elif len(feat_label) == 3:
        node_feat, edge_feat, labels = feat_label
    return node_feat, edge_feat, labels


def meta_gradient_step(args, data_generator, model, optimiser, input_data, inner_train_steps, inner_lr,
                       train, batch_n, device):
    create_graph = True and train
    meta_train_graphs, meta_test_graphs = input_data
    task_losses = []
    task_acc = []
    task_ae_losses = []

    if train:
        model.train()
    else:
        model.eval()

    # Hence when we iterate over the first dimension we are iterating through the meta batches
    for meta_batch in range(batch_n):

        train_graphs, eval_graphs = meta_train_graphs[meta_batch], meta_test_graphs[meta_batch]

        train_node_feat, train_edge_feat, train_labels = judge_feat_label(
            data_generator.PrepareFeatureLabel(train_graphs, -1))

        eval_node_feat, eval_edge_feat, eval_labels = judge_feat_label(
            data_generator.PrepareFeatureLabel(eval_graphs, -1))

        fast_weights = OrderedDict(model.named_parameters())

        # Train the model for `inner_train_steps` iterations
        if args.use_maml == 1:
            for inner_batch in range(inner_train_steps):
                # Perform update of model weights
                train_output = model.functional_forward(train_graphs, train_node_feat, train_edge_feat, fast_weights)
                loss_train = F.nll_loss(train_output, train_labels)
                # acc_train=accuracy(output[idx_train], labels[idx_train])
                gradients = torch.autograd.grad(loss_train, fast_weights.values(), create_graph=create_graph)
                # print(inner_batch, loss_train, acc_train)
                # Update weights manually
                fast_weights = OrderedDict(
                    (name, param - inner_lr * grad)
                    for ((name, param), grad) in zip(fast_weights.items(), gradients)
                )

        if args.use_proto == 1:
            prototype = []

            output_repr_train = model.functional_forward(train_graphs, train_node_feat, train_edge_feat, fast_weights,
                                                         eval=True)

            for class_graph_id in range(data_generator.num_class):
                sel_id = [idx for idx, x in enumerate(train_graphs) if x.label == class_graph_id]
                prototype.append(torch.mean(output_repr_train[sel_id], dim=0))

            prototype = torch.stack(prototype, dim=0)

            eval_output_emb = model.functional_forward(eval_graphs, eval_node_feat, eval_edge_feat, fast_weights,
                                                       eval=True)

            eval_output = []

            for class_id in range(data_generator.num_class):
                eval_output.append(torch.sum(-(eval_output_emb - prototype[class_id]).pow(2), dim=1))
                # eval_output.append(torch.tanh(torch.mm(eval_output_emb, torch.unsqueeze(prototype[class_id], dim=1))))

            eval_output = torch.squeeze(torch.stack(eval_output).transpose(0, 1))

            # test_output = torch.mm(output_eval_emb, torch.transpose(prototype, 0, 1))

            eval_output = F.log_softmax(eval_output, dim=1)

        elif args.use_proto == 0:
            eval_output = model.functional_forward(eval_graphs, eval_node_feat, eval_edge_feat, fast_weights)

        # Do a pass of the model on the validation data from the current task
        loss = F.nll_loss(eval_output, eval_labels)
        acc = accuracy(eval_output, eval_labels)
        task_acc.append(acc)
        task_losses.append(loss)

    meta_batch_loss = torch.stack(task_losses).mean()
    meta_batch_acc = torch.stack(task_acc).mean()
    meta_batch_ci = 1.96 * torch.stack(task_acc).std() / np.sqrt(batch_n)

    if train:
        optimiser.zero_grad()
        meta_batch_loss.backward()
        optimiser.step()

    return meta_batch_loss, meta_batch_acc, meta_batch_ci


def fine_tune(args, data_generator, model, optimiser, input_data, inner_train_steps, inner_lr,
                       train, batch_n, device):
    create_graph = True and train
    meta_train_graphs, meta_test_graphs = input_data
    task_losses = []
    task_acc = []
    task_ae_losses = []

    if train:
        model.train()
    else:
        model.eval()


    train_graphs, eval_graphs = meta_train_graphs, meta_test_graphs

    train_node_feat, train_edge_feat, train_labels = judge_feat_label(
        data_generator.PrepareFeatureLabel(train_graphs, -1))

    eval_node_feat, eval_edge_feat, eval_labels = judge_feat_label(
        data_generator.PrepareFeatureLabel(eval_graphs, -1))

    fast_weights = OrderedDict(model.named_parameters())

    train_output = model.functional_forward(train_graphs, train_node_feat, train_edge_feat, fast_weights)
    eval_output = model.functional_forward(eval_graphs, eval_node_feat, eval_edge_feat, fast_weights)


    loss = F.nll_loss(eval_output, eval_labels)
    acc = accuracy(eval_output, eval_labels)
    task_acc.append(acc)
    task_losses.append(loss)

    meta_batch_loss = torch.stack(task_losses).mean()
    meta_batch_acc = torch.stack(task_acc).mean()
    meta_batch_ci = 1.96 * torch.stack(task_acc).std() / np.sqrt(batch_n)

    train_loss = F.nll_loss(train_output, train_labels)
    if train:
        optimiser.zero_grad()
        train_loss.backward()
        optimiser.step()

    return meta_batch_loss, meta_batch_acc, meta_batch_ci


def meta_gradient_step_matching(args, data_generator, model, optimiser, input_data, inner_train_steps, inner_lr,
                                train, batch_n, device):
    create_graph = True and train
    meta_train_graphs, meta_test_graphs = input_data
    task_losses = []
    task_acc = []
    task_ae_losses = []

    if train:
        model.train()
    else:
        model.eval()

    # Hence when we iterate over the first dimension we are iterating through the meta batches
    for meta_batch in range(batch_n):

        train_graphs, eval_graphs = meta_train_graphs[meta_batch], meta_test_graphs[meta_batch]

        fast_weights = OrderedDict(model.named_parameters())

        prototype = []

        # output_repr_train = model.functional_forward(train_graphs, train_node_feat, train_edge_feat, fast_weights, eval=True)

        for class_graph_id in range(data_generator.num_class):

            train_node_feat, train_edge_feat, train_labels = judge_feat_label(
                data_generator.PrepareFeatureLabel(train_graphs, class_graph_id))

            eval_node_feat, eval_edge_feat, eval_labels = judge_feat_label(
                data_generator.PrepareFeatureLabel(eval_graphs, -1))

            sel_train_graphs = [x for idx, x in enumerate(train_graphs) if x.label == class_graph_id]

            prototype.append(model(eval_graphs, sel_train_graphs, eval_node_feat, train_node_feat,
                                                         eval_edge_feat, train_edge_feat, fast_weights, class_graph_id, eval=True))

        eval_output = torch.squeeze(torch.transpose(torch.stack(prototype, dim=0), 0, 1))

        # ipdb.set_trace()

        eval_output = F.log_softmax(eval_output, dim=1)

        # Do a pass of the model on the validation data from the current task
        loss = F.nll_loss(eval_output, eval_labels)
        acc = accuracy(eval_output, eval_labels)
        task_acc.append(acc)
        task_losses.append(loss)

    meta_batch_loss = torch.stack(task_losses).mean()
    meta_batch_acc = torch.stack(task_acc).mean()
    meta_batch_ci = 1.96 * torch.stack(task_acc).std() / np.sqrt(batch_n)

    if train:
        optimiser.zero_grad()
        meta_batch_loss.backward()
        optimiser.step()

    return meta_batch_loss, meta_batch_acc, meta_batch_ci
