import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool, MessagePassing, inits
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.datasets import Planetoid, KarateClub
from torch_geometric.utils import add_self_loops, degree


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer=2, drop_ratio=0, pool='mean'):
        super().__init__()
        GraphConv = GCNConv
        if num_layer < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(num_layer))
        elif num_layer == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hidden_dim), GraphConv(hidden_dim, output_dim)])
        else:
            layers = [GraphConv(input_dim, hidden_dim)]
            for i in range(num_layer - 2):
                layers.append(GraphConv(hidden_dim, hidden_dim))
            layers.append(GraphConv(hidden_dim, output_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

        self.drop_ratio = drop_ratio
        # Different kind of graph pooling
        if pool == "sum":
            self.pool = global_add_pool
        elif pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        # elif pool == "attention":
        #     self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, data, prompt=None, prompt_type=None, pooling=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h_list = [x]
        for idx, conv in enumerate(self.conv_layers[0:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_ratio, training=self.training)
            h_list.append(x)
        x = self.conv_layers[-1](x, edge_index)
        h_list.append(x)

        node_emb = h_list[-1]
        if pooling:
            # Subgraph pooling to obtain the graph embeddings
            graph_emb = self.pool(node_emb, batch.long())
        else:
            # Extract the embedding of the target nodes as the graph embeddings
            graph_emb = node_emb[data.ptr[:-1] + data.target_node]
        return graph_emb

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


class GCNConvWithEdgeAttr(MessagePassing):
    def __init__(self, input_dim, output_dim):
        super(GCNConvWithEdgeAttr, self).__init__(aggr='add')
        self.lin = Linear(input_dim, output_dim, bias=False, weight_initializer='glorot')
        self.bias = nn.Parameter(torch.empty(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        inits.zeros(self.bias)

    def forward(self, x, edge_index, edge_prompt=False):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm, edge_attr=edge_prompt)

        out = out + self.bias
        return out

    def message(self, x_j, norm, edge_attr):
        if edge_attr is not False:
            return norm.view(-1, 1) * self.lin(x_j + edge_attr)
        else:
            return norm.view(-1, 1) * self.lin(x_j)


class CustomGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_ratio=0.5):
        super(CustomGCN, self).__init__()
        self.conv1 = GCNConvWithEdgeAttr(input_dim, hidden_dim)
        self.conv2 = GCNConvWithEdgeAttr(hidden_dim, output_dim)
        self.drop_ratio = drop_ratio

    def forward(self, data, prompt_type=None, prompt=None, pooling=False):
        assert pooling in ['mean', 'target', False]
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_ratio, training=self.training)
        x = self.conv2(x, edge_index)

        if pooling == 'mean':
            # Subgraph pooling to obtain the graph embeddings
            graph_emb = global_mean_pool(x, batch.long())
            return graph_emb
        if pooling == 'target':
            # Extract the embedding of target nodes as the graph embeddings
            graph_emb = x[data.ptr[:-1] + data.target_node]
            return graph_emb

        return x


class GCN_DGI(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_ratio=0.5):
        super(GCN_DGI, self).__init__()
        self.conv1 = GCNConvWithEdgeAttr(input_dim, hidden_dim)
        self.conv2 = GCNConvWithEdgeAttr(hidden_dim, output_dim)
        self.drop_ratio = drop_ratio

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_ratio, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class OfficialGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_ratio=0):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.drop_ratio = drop_ratio

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_ratio, training=self.training)
        x = self.conv2(x, edge_index)

        return x


if __name__ == '__main__':
    dataset = KarateClub()
    # device = torch.device(f'cuda:2' if torch.cuda.is_available() else 'cpu')
    # dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=f'./data/ogb')
    data = dataset[0]
    # gcn1 = GCNConv(in_channels=data.num_node_features, out_channels=3, add_self_loops=True, normalize=True)
    # gcn2 = CustomGCNConv(in_channels=data.num_node_features, out_channels=3)
    # for para1, para2 in zip(gcn1.parameters(), gcn2.parameters()):
    #     para2.data = deepcopy(para1.data)
    # emb1 = gcn1(data.x, data.edge_index)
    # emb2 = gcn2(data.x, data.edge_index)
    # print(1)
    edge_prompt = nn.Parameter(torch.ones([1, dataset.num_node_features]))

    gcn1 = OfficialGCN(dataset.num_node_features, 12, dataset.num_classes)
    gcn2 = CustomGCN(dataset.num_node_features, 12, dataset.num_classes)
    for para1, para2 in zip(gcn1.parameters(), gcn2.parameters()):
        para1.data = deepcopy(para2.data)

    emb1 = gcn1(data)
    emb2 = gcn2(data, edge_prompt=edge_prompt)
    print(1)
