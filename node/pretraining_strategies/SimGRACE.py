import random
import numpy as np
from copy import deepcopy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from load_data import load_node_data, NodePretrain
from model import CustomGCN, GCN_DGI


def perturb(gnn, device, eta=1.0):
    vice_gnn = deepcopy(gnn)
    for para in vice_gnn.parameters():
        # para.data = para.data + eta * torch.normal(0, torch.ones_like(para.data) * para.data.std()).to(device)
        std = para.data.std() if para.data.numel() > 1 else torch.tensor(1.0)
        para.data += eta * torch.normal(0, torch.ones_like(para.data) * std).to(device)

    return vice_gnn


class SimGRACE(nn.Module):
    def __init__(self, dataset_name, gnn_type, hidden_dim, batch_size, device, seed, logger):
        super(SimGRACE, self).__init__()
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

        self.projection_head = torch.nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                   nn.ReLU(inplace=True),
                                                   nn.Linear(self.hidden_dim, self.hidden_dim)).to(self.device)

    def compute_loss(self, x, x_aug, tau=0.2):
        batch_size = x.shape[0]
        cosine_sim = F.cosine_similarity(x.unsqueeze(1), x_aug.unsqueeze(0), dim=2)
        sim_matrix = torch.exp(cosine_sim / tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

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
                x = self.projection_head(self.gnn(data, pooling='mean'))
                vice_gnn = perturb(self.gnn, self.device)
                x_aug = self.projection_head(vice_gnn(data, pooling='mean'))
                loss = self.compute_loss(x, x_aug.detach())
                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())

            if epoch % 1 == 0:
                log_info = ''.join(['| Epoch: [{:4d} / {:4d}] '.format(epoch, epochs),
                                    '| average_loss: {:7.5f} |'.format(np.mean(total_loss))]
                                   )
                self.logger.info(log_info)
        t2 = time.time()
        pretrained_gnn_file = f'./pretrained_gnns/{self.dataset_name}_SimGRACE_{self.gnn_type}_{self.seed}.pth'
        torch.save(self.gnn.state_dict(), pretrained_gnn_file)
        pretrained_ph_file = f'./pretrained_gnns/{self.dataset_name}_SimGRACE_ph_{self.seed}.pth'
        torch.save(self.projection_head.state_dict(), pretrained_ph_file)
        log_info = ''.join(['Pretraining time: {:.2f} seconds | '.format(t2 - t1),
                           f'pretrained model saved in {pretrained_gnn_file}'])
        self.logger.info(log_info)
