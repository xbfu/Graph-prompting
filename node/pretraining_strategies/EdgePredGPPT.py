import random
import numpy as np
from copy import deepcopy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import batched_negative_sampling

from load_data import load_node_data, NodePretrain
from model import CustomGCN, GCN_DGI


def mask_edges(edge_index, k):
    num_edges = edge_index.shape[1]
    mask = torch.ones(num_edges, dtype=torch.bool)
    perm = torch.randperm(num_edges)[:k]
    mask[perm] = False

    masked_edge_index = edge_index[:, mask]
    masked_edges = edge_index[:, ~mask]

    return masked_edge_index, masked_edges


class EdgePredGPPT(nn.Module):
    def __init__(self, dataset_name, gnn_type, hidden_dim, batch_size, device, seed, logger):
        super(EdgePredGPPT, self).__init__()
        self.dataset_name = dataset_name
        self.gnn_type = gnn_type
        self.hidden_dim = hidden_dim
        self.device = device
        self.seed = seed
        self.logger = logger
        if dataset_name in ['Cora', 'CiteSeer', 'PubMed', 'Flickr', 'ogbn-arxiv', 'Actor', 'photo']:
            data, input_dim, output_dim = load_node_data(dataset_name, data_folder='./data')
            print('Dataset: ', dataset_name, '| num_features: ', input_dim, '| num_classes: ', output_dim)
            self.graph_list = NodePretrain(data, batch_size)
            self.input_dim = input_dim
            self.output_dim = output_dim
        else:
            raise ValueError('Error: invalid dataset name! Supported datasets: [Cora, CiteSeer, PubMed, Flickr, Actor, ogbn-arxiv]')

        self.initialize_model()

    def initialize_model(self):
        if self.gnn_type == 'GCN':
            self.gnn = CustomGCN(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.hidden_dim)
        else:
            raise ValueError(f"Error: invalid GNN type! Suppported GNNs: [GCN, GIN]")

        print(self.gnn)
        self.gnn.to(self.device)

        self.projection_head = torch.nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                                                   nn.ReLU(inplace=True),
                                                   nn.Linear(self.hidden_dim, 1)).to(self.device)

    def pretrain(self, batch_size, lr, decay, epochs):
        loader = DataLoader(self.graph_list, batch_size=batch_size, shuffle=True)
        learnable_parameters = list(self.gnn.parameters()) + list(self.projection_head.parameters())
        optimizer = torch.optim.Adam(learnable_parameters, lr=lr, weight_decay=decay)
        print('Start training')
        t1 = time.time()
        for epoch in range(1, 1 + epochs):
            total_loss = []
            self.gnn.train()
            for i, data in enumerate(loader):
                optimizer.zero_grad()
                data = data.to(self.device)
                neg_edge_index = batched_negative_sampling(data.edge_index, data.batch, num_neg_samples=1)
                # neg_edge_index = negative_sampling(data.edge_index, data.x.shape[0], num_neg_samples=batch_size)
                masked_edge_index, pos_edge_index = mask_edges(data.edge_index, k=batch_size)
                edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
                edge_label = torch.cat([torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0)
                data.edge_index = masked_edge_index
                emb = self.gnn(data, pooling=False)
                edge_emb = torch.cat([emb[edge_index[0]], emb[edge_index[1]]], dim=1)

                edge_pred = self.projection_head(edge_emb)
                loss = F.binary_cross_entropy_with_logits(edge_pred.squeeze(), edge_label.to(self.device))
                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())

            if epoch % 1 == 0:
                log_info = ''.join(['| Epoch: [{:4d} / {:4d}] '.format(epoch, epochs),
                                    '| average_loss: {:7.5f} |'.format(np.mean(total_loss))]
                                   )
                self.logger.info(log_info)
        t2 = time.time()
        pretrained_gnn_file = f'./pretrained_gnns/{self.dataset_name}_EdgePredGPPT_{self.gnn_type}_{self.seed}.pth'
        torch.save(self.gnn.state_dict(), pretrained_gnn_file)
        pretrained_ph_file = f'./pretrained_gnns/{self.dataset_name}_EdgePredGPPT_ph_{self.seed}.pth'
        torch.save(self.projection_head.state_dict(), pretrained_ph_file)
        log_info = ''.join(['Pretraining time: {:.2f} seconds | '.format(t2 - t1),
                           f'pretrained model saved in {pretrained_gnn_file}'])
        self.logger.info(log_info)


def run(dataset_name, gnn_type, hidden_dim, batch_size, device, seed, lr=0.001, decay=0, epochs=20):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False     # Disable CUDNN auto-tuning

    pt = EdgePredGPPT(dataset_name, gnn_type, hidden_dim, batch_size, device, seed)
    pt.pretrain(batch_size, lr, decay, epochs)


if __name__ == '__main__':
    gnn_type = 'GCN'
    hidden_dim = 128
    batch_size = 32
    gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # dataset_list = ['Cora', 'CiteSeer', 'PubMed', 'Actor', 'ogbn-arxiv', 'Flickr']
    dataset_list = ['ogbn-arxiv']
    # seed_list = [0, 1, 2, 3, 4]
    seed_list = [0]
    for dataset_name in dataset_list:
        for seed in seed_list:
            run(dataset_name, gnn_type, hidden_dim, batch_size, device, seed)
