import argparse
import importlib
import json

import numpy as np
import glob
import torch
import torch.nn as nn


def get_criterion(task):
    """
    Parameters
    ----------
    task : str
        Name of the task.
    Returns
    -------
    criterion : torch.nn.modules._Loss
        Loss function for the task.
    """
    if task == 'link_prediction':
        # Pos weight to balance dataset without oversampling
        criterion = nn.BCELoss()#pos_weight=torch.FloatTensor([7.]))

    return criterion


def get_dataset(args, setPath=None, add_self_edges=False):
    """
    Parameters
    ----------
    args : tuple
        Tuple of task, dataset name and other arguments required by the dataset constructor.
    add_self_edges: boolean
        True for adding self edges, otherwise False.
    Returns
    -------
    dataset : torch.utils.data.Dataset
        The dataset.
    """
    datasets = []
    mode, num_layers = args

    train_paths = []
    test_paths = []
    val_paths = []

    train_glob = glob.glob(
        f'datasets/dataset/Train/*')
    test_glob = glob.glob(
        f'datasets/dataset/Test/*')
    val_glob = glob.glob(
        f'datasets/dataset/Val/*')


    train_glob = sorted(train_glob)
    test_glob = sorted(test_glob)
    val_glob = sorted(val_glob)

    for i in range(0, len(train_glob), 2):
        train_paths.append([train_glob[i].split('/')[-1]
        .replace('_delaunay_orig_forGraphSAGE_edges.csv', ''), train_glob[i], train_glob[i+1]])

    for i in range(0, len(test_glob), 2):
        test_paths.append([test_glob[i].split('/')[-1]
        .replace('_delaunay_orig_forGraphSAGE_edges.csv', ''), test_glob[i], test_glob[i+1]])
    
    for i in range(0, len(val_glob), 2):
        val_paths.append([val_glob[i].split('/')[-1]
        .replace('_delaunay_orig_forGraphSAGE_edges.csv', ''), val_glob[i], val_glob[i+1]])


    if setPath == None:
        if mode == 'train':
            for path in train_paths:
                class_attr = getattr(importlib.import_module('datasets.link_prediction'), 'KIGraphDatasetGCN')
                dataset = class_attr(path, mode, num_layers, add_self_edges)
                datasets.append(dataset)
        elif mode == 'val':
            for path in val_paths:
                class_attr = getattr(importlib.import_module('datasets.link_prediction'), 'KIGraphDatasetGCN')
                dataset = class_attr(path, mode, num_layers, add_self_edges)
                datasets.append(dataset)
        elif mode == 'test':
            for path in test_paths:
                class_attr = getattr(importlib.import_module('datasets.link_prediction'), 'KIGraphDatasetGCN')
                dataset = class_attr(path, mode, num_layers, add_self_edges)
                datasets.append(dataset)
    else:
        class_attr = getattr(importlib.import_module('datasets.link_prediction'), 'KIGraphDatasetGCN')
        dataset = class_attr(setPath, mode, num_layers)
        datasets.append(dataset)

    return datasets


def get_fname(config):
    """
    Parameters
    ----------
    config : dict
        A dictionary with all the arguments and flags.
    Returns
    -------
    fname : str
        The filename for the saved model.
    """
    model = config['model']
    fname = f"{model}.pth"

    return fname


def normalize_edge_features_rows(edge_features):
    """
    Parameters
    ----------
    edge_features : numpy array
        3d numpy array (P x N x N).
        edge_features[p, i, j] is the jth feature of node i in pth channel
    Returns
    -------
    edge_features_normed : numpy array
        normalized edge_features.
    """
    deno = np.sum(np.abs(edge_features), axis=2, keepdims=True)
    return np.divide(edge_features, deno, where = deno != 0)


def concat_node_representations_double(features, edges, device="cpu"):
    """
    Parameters
    ----------
    features : torch.Tensor
        features[i] is the representation of node i.
    device : string
        'cpu' or 'cuda:0'. Default: 'cpu'.
    Returns
    ----------
    out1: torch.Tensor
        Concatinated features.
    out2: torch.Tensor
        Concatinated features.
    """
    out1 = torch.FloatTensor().to(device)
    out2 = torch.FloatTensor().to(device)
    for node1, node2 in edges:
        node12 = torch.cat((features[node1], features[node2])).reshape(1, -1)
        node21 = torch.cat((features[node2], features[node1])).reshape(1, -1)
        out1 = torch.cat((out1, node12), dim=0)
        out2 = torch.cat((out2, node21), dim=0)

    return out1, out2


def parse_args():
    """
    Returns
    -------
    config : dict
        A dictionary with the required arguments and flags.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--json', type=str, default='config.json',
                        help='path to json file with arguments, default: config.json')

    parser.add_argument('--stats_per_batch', type=int, default=16,
                        help='print loss and accuracy after how many batches, default: 16')
    
    parser.add_argument('--saved_models_dir', type=str,
                        help='path to save models')

    parser.add_argument('--task', type=str,
                        choices=['unsupervised', 'link_prediction'],
                        default='link_prediction',
                        help='type of task, default=link_prediction')
    parser.add_argument('--mode', type=str,
                        choices=['train', 'val', 'test'],
                        default='train',
                        help='running mode, default=train')

    parser.add_argument('--cuda', action='store_true',
                        help='whether to use GPU, default: False')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout out, currently only for GCN, default: 0.5')
    parser.add_argument('--hidden_dims', type=int, nargs="*",
                        help='dimensions of hidden layers, length should be equal to num_layers, specify through config.json')
    parser.add_argument('--out_dim', type=int, default=1,
                        help='dimension of the model\'s output layer, default=1')                    
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='number of neighbors to sample, default=-1')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='training batch size, default=32')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of training epochs, default=2')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate, default=1e-4')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay, default=5e-4')
    parser.add_argument('--threshold', type=float, default=0.90,
                        help='threshold, default=0.90')
    parser.add_argument('--model', type=str, default='egnnc',
                        help='model name, default: egnnc')

    args = parser.parse_args()
    config = vars(args)
    if config['json']:
        with open(config['json']) as f:
            json_dict = json.load(f)
            config.update(json_dict)

            for (k, v) in config.items():
                if config[k] == 'True':
                    config[k] = True
                elif config[k] == 'False':
                    config[k] = False

    config['num_layers'] = len(config['hidden_dims']) + 1

    print('--------------------------------')
    print('Config:')
    for (k, v) in config.items():
        print("    '{}': '{}'".format(k, v))
    print('--------------------------------')

    return config
