import numpy as np
from copy import deepcopy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import structured_negative_sampling

from load_data import load_graph_data
from model import CustomGIN


def to_numpy(tensor, digits: int = 4):
    return np.around(tensor.cpu().detach().numpy(), digits)


class EdgePredGraphPrompt(nn.Module):
    def __init__(self, dataset_name, gnn_type, num_layer, hidden_dim, batch_size, device, seed, logger):
        super(EdgePredGraphPrompt, self).__init__()
        self.dataset_name = dataset_name
        self.gnn_type = gnn_type
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.device = device
        self.seed = seed
        self.logger = logger
        if dataset_name in ['ENZYMES', 'PROTEINS', 'BACE', 'BBBP', 'DD', 'MUTAG', 'NCI1', 'Mutagenicity', 'NCI109']:
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

        self.projection_head = torch.nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                   nn.ReLU(inplace=True),
                                                   nn.Linear(self.hidden_dim, self.hidden_dim)).to(self.device)

    def pretrain(self, batch_size, lr, decay, epochs, tau=0.2):
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
                emb = self.gnn(data, pooling=False)
                v, a, b = structured_negative_sampling(data.edge_index,
                                                       num_nodes=data.x.shape[0],
                                                       contains_neg_self_loops=False)
                sampled_nodes = torch.randperm(data.x.shape[0])[:batch_size]
                s_v = self.projection_head(emb[v[sampled_nodes]])
                s_a = self.projection_head(emb[a[sampled_nodes]])
                s_b = self.projection_head(emb[b[sampled_nodes]])
                pos_sim, neg_sim = F.cosine_similarity(s_v, s_a), F.cosine_similarity(s_v, s_b)
                numerator = torch.exp(pos_sim / tau)
                denominator = numerator + torch.exp(neg_sim / tau)
                loss = - torch.log(numerator / denominator).mean()
                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())

            if epoch % 1 == 0:
                log_info = ''.join(['| Epoch: [{:4d} / {:4d}] '.format(epoch, epochs),
                                    '| average_loss: {:7.5f} |'.format(np.mean(total_loss))]
                                   )
                self.logger.info(log_info)

        t2 = time.time()
        pretrained_gnn_file = f'./pretrained_gnns/{self.dataset_name}_EdgePredGraphPrompt_{self.gnn_type}_{self.seed}.pth'
        torch.save(self.gnn.state_dict(), pretrained_gnn_file)
        pretrained_ph_file = f'./pretrained_gnns/{self.dataset_name}_EdgePredGraphPrompt_ph_{self.seed}.pth'
        torch.save(self.projection_head.state_dict(), pretrained_ph_file)
        log_info = ''.join(['Pretraining time: {:.2f} seconds | '.format(t2 - t1),
                            f'pretrained model saved in {pretrained_gnn_file}'])
        self.logger.info(log_info)
