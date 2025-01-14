import random
import numpy as np
from copy import deepcopy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn.inits import reset, uniform

from get_args import get_args
from load_data import load_node_data, NodePretrain
from model import CustomGCN, GCN_DGI


def to_numpy(tensor, digits: int = 4):
    return np.around(tensor.cpu().detach().numpy(), digits)


def corruption(x):
    return x[torch.randperm(x.size(0), device=x.device)]


class DeepGraphInfomax(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DeepGraphInfomax, self).__init__()
        self.gnn = GCN_DGI(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.w = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        reset(self.gnn)
        uniform(hidden_dim, self.w)

    def discriminate(self, z, summary, batch, sigmoid):
        value = torch.sum((z @ self.w) * summary[batch], dim=1)
        # value = torch.matmul(z, torch.matmul(self.w, summary))
        return torch.sigmoid(value) if sigmoid else value

    def forward(self, data):
        pos_z = self.gnn(data.x, data.edge_index)
        summary = torch.sigmoid(torch.mean(pos_z, dim=0))
        corrupted_x = corruption(data.x)
        neg_z = self.gnn(corrupted_x, data.edge_index)
        pos_loss = - torch.log(self.discriminate(pos_z, summary, data.batch, sigmoid=True) + 1e-15).mean()
        neg_loss = - torch.log(1 - self.discriminate(neg_z, summary, data.batch, sigmoid=True) + 1e-15).mean()
        return pos_loss + neg_loss


class DGI(nn.Module):
    def __init__(self, dataset_name, gnn_type, hidden_dim, batch_size, device, seed, logger):
        super(DGI, self).__init__()
        self.dataset_name = dataset_name
        self.gnn_type = gnn_type
        self.hidden_dim = hidden_dim
        self.device = device
        self.seed = seed
        self.logger = logger
        if dataset_name in ['Cora', 'CiteSeer', 'PubMed', 'Flickr', 'ogbn-arxiv', 'Actor']:
            self.data, input_dim, output_dim = load_node_data(dataset_name, data_folder='./data')
            print('Dataset: ', dataset_name, '| num_features: ', input_dim, '| num_classes: ', output_dim)
            self.data = self.data.to(device)
            self.input_dim = input_dim
            self.output_dim = output_dim
        else:
            raise ValueError('Error: invalid dataset name! Supported datasets: [Cora, CiteSeer, PubMed, Flickr, Actor, ogbn-arxiv]')

        self.model = DeepGraphInfomax(input_dim, hidden_dim).to(device)

    def pretrain(self, batch_size, lr, decay, epochs):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)
        print('Start training')
        t1 = time.time()
        self.model.train()
        for epoch in range(1, 1 + epochs):
            total_loss = []
            for i in range(80):
                optimizer.zero_grad()
                loss = self.model(self.data)
                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())

            if epoch % 1 == 0:
                log_info = ''.join(['| Epoch: [{:4d} / {:4d}] '.format(epoch, epochs),
                                    '| average_loss: {:7.5f} |'.format(np.mean(total_loss))]
                                   )
                self.logger.info(log_info)
        t2 = time.time()
        pretrained_gnn_file = f'./pretrained_gnns/{self.dataset_name}_DGI_{self.gnn_type}_0.pth'
        torch.save(self.model.gnn.state_dict(), pretrained_gnn_file)
        log_info = ''.join(['Pretraining time: {:.2f} seconds | '.format(t2 - t1),
                           f'pretrained model saved in {pretrained_gnn_file}'])
        self.logger.info(log_info)


if __name__ == '__main__':
    args = get_args()
    print(' | Dataset:            ', args.dataset_name,
          '\n | epochs:             ', args.epochs,
          '\n | hidden_dim:         ', args.hidden_dim,
          '\n | batch_size:         ', args.batch_size,
          '\n | device_id:          ', args.device,
          '\n | seed:               ', args.seed,
          )

    pt = DGI(args.dataset_name, args.gnn_type, args.hidden_dim, args.batch_size, args.device, args.seed)
    pt.pretrain(args.batch_size, args.lr, args.decay, args.epochs)

