import random
import logging
import numpy as np
import torch
from pretraining_strategies import EdgePredGPPT, SimGRACE, EdgePredGraphPrompt
from logger import Logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(dataset_name, pretrain_task, gnn_type, hidden_dim, batch_size, device, seed, lr=0.001, decay=0, epochs=200):
    filename = f'log/{dataset_name}_{pretrain_task}_{gnn_type}_{seed}.log'
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger = Logger(filename, formatter)

    set_random_seed(seed)

    if pretrain_task == 'EdgePredGPPT':
        pt = EdgePredGPPT.EdgePredGPPT(dataset_name, gnn_type, hidden_dim, batch_size, device, seed, logger)
        pt.pretrain(batch_size, lr, decay, epochs)
    elif pretrain_task == 'SimGRACE':
        pt = SimGRACE.SimGRACE(dataset_name, gnn_type, hidden_dim, batch_size, device, seed, logger)
        pt.pretrain(batch_size, lr, decay, epochs)
    elif pretrain_task == 'EdgePredGraphPrompt':
        pt = EdgePredGraphPrompt.EdgePredGraphPrompt(dataset_name, gnn_type, hidden_dim, batch_size, device, seed, logger)
        pt.pretrain(batch_size, lr, decay, epochs)


if __name__ == '__main__':
    gnn_type = 'GCN'
    hidden_dim = 128
    batch_size = 128
    gpu_id = 0
    pretrain_list = ['SimGRACE', 'EdgePredGPPT', 'EdgePredGraphPrompt']
    # pretrain_list = ['SimGRACE']
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # dataset_list = ['Cora', 'CiteSeer', 'PubMed', 'Actor', 'ogbn-arxiv', 'Flickr']
    dataset_list = ['PubMed']
    seed = 0
    for dataset_name in dataset_list:
        for pretrain_task in pretrain_list:
            print('pretrain_task: ', pretrain_task, ' | hidden_dim: ', hidden_dim, ' | batch_size: ', batch_size)
            run(dataset_name, pretrain_task, gnn_type, hidden_dim, batch_size, device, seed)
