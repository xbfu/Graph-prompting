import random
import numpy as np
from copy import deepcopy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import batched_negative_sampling

from load_data import load_graph_data
from model import CustomGIN


def mask_edges(edge_index, k):
    num_edges = edge_index.shape[1]
    mask = torch.ones(num_edges, dtype=torch.bool)
    perm = torch.randperm(num_edges)[:k]
    mask[perm] = False

    masked_edge_index = edge_index[:, mask]
    masked_edges = edge_index[:, ~mask]

    return masked_edge_index, masked_edges


class EdgePredGPPT(nn.Module):
    def __init__(self, dataset_name, gnn_type, num_layer, hidden_dim, batch_size, device, seed, logger):
        super(EdgePredGPPT, self).__init__()
        self.dataset_name = dataset_name
        self.gnn_type = gnn_type
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.device = device
        self.seed = seed
        self.logger = logger
        if dataset_name in ['ENZYMES', 'COX2', 'PROTEINS', 'BZR', 'DD', 'NCI1', 'NCI109', 'Mutagenicity']:
            data, input_dim, output_dim = load_graph_data(dataset_name, data_folder='./data')
            self.graph_list = [graph for graph in data]
            self.input_dim = input_dim
            self.output_dim = output_dim
        else:
            raise ValueError('Error: invalid dataset name! Supported datasets: [ENZYMES, COX2, PROTEINS]')

        self.initialize_model()

    def initialize_model(self):
        if self.gnn_type == 'GIN':
            self.gnn = CustomGIN(num_layer=self.num_layer, input_dim=self.input_dim, hidden_dim=self.hidden_dim)
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
