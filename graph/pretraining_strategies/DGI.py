import random
import numpy as np
from copy import deepcopy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn.inits import reset, uniform
from torch_geometric.nn import global_mean_pool
from get_args import get_args
from load_data import load_graph_data
from model import CustomGIN, GIN_DGI


def to_numpy(tensor, digits: int = 4):
    return np.around(tensor.cpu().detach().numpy(), digits)


def corruption(x):
    return x[torch.randperm(x.size(0), device=x.device)]


class DeepGraphInfomax(nn.Module):
    def __init__(self, num_layer, input_dim, hidden_dim):
        super(DeepGraphInfomax, self).__init__()
        self.gnn = GIN_DGI(num_layer=num_layer, input_dim=input_dim, hidden_dim=hidden_dim)
        self.w = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        reset(self.gnn)
        uniform(hidden_dim, self.w)

    def discriminate(self, z, summary, batch, sigmoid: bool = True):
        # summary = summary.t() if summary.dim() > 1 else summary
        # value = torch.matmul(z, torch.matmul(self.w, summary))
        value = torch.sum((z @ self.w) * summary[batch], dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward(self, data):
        node_emb = self.gnn(data.x, data.edge_index, pooling=False)
        # summary = torch.sigmoid(torch.mean(pos_z, dim=0))
        summary = torch.sigmoid(global_mean_pool(node_emb, data.batch))
        neg_summary = summary[torch.randperm(summary.size(0), device=summary.device)]
        pos_loss = -torch.log(self.discriminate(node_emb, summary, data.batch, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.discriminate(node_emb, neg_summary, data.batch, sigmoid=True) + 1e-15).mean()
        return pos_loss + neg_loss


class DGI(nn.Module):
    def __init__(self, dataset_name, gnn_type, num_layer, hidden_dim, batch_size, device, seed, logger):
        super(DGI, self).__init__()
        self.dataset_name = dataset_name
        self.gnn_type = gnn_type
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.device = device
        self.seed = seed
        self.logger = logger
        if dataset_name in ['ENZYMES', 'COX2', 'PROTEINS', 'BZR', 'DD']:
            data, input_dim, output_dim = load_graph_data(dataset_name, data_folder='./data')
            self.graph_list = [graph for graph in data]
            self.input_dim = input_dim
            self.output_dim = output_dim
        else:
            raise ValueError('Error: invalid dataset name! Supported datasets: [ENZYMES, COX2, PROTEINS]')

        self.initialize_model()

    def initialize_model(self):
        self.model = DeepGraphInfomax(self.num_layer, self.input_dim, self.hidden_dim).to(self.device)

    def pretrain(self, batch_size, lr, decay, epochs):
        loader = DataLoader(self.graph_list, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)
        print('Start training')
        t1 = time.time()
        for epoch in range(1, 1 + epochs):
            total_loss = []
            self.model.train()
            for i, data in enumerate(loader):
                optimizer.zero_grad()
                data = data.to(self.device)
                loss = self.model(data)
                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())

            if epoch % 1 == 0:
                log_info = ''.join(['| Epoch: [{:4d} / {:4d}] '.format(epoch, epochs),
                                    '| average_loss: {:7.5f} |'.format(np.mean(total_loss))]
                                   )
                self.logger.info(log_info)
        t2 = time.time()
        pretrained_gnn_file = f'./pretrained_gnns/{self.dataset_name}_DGI_{self.gnn_type}_{self.seed}.pth'
        torch.save(self.model.gnn.state_dict(), pretrained_gnn_file)
        log_info = ''.join(['Pretraining time: {:.2f} seconds | '.format(t2 - t1),
                           f'pretrained model saved in {pretrained_gnn_file}'])
        self.logger.info(log_info)
