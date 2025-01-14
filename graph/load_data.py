import random
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import Planetoid, KarateClub, TUDataset, MoleculeNet
from torch_geometric.loader import DataLoader

empty_graphs = [59, 61, 391, 614, 642, 645, 646, 647, 648, 649, 685]


def load_graph_data(dataset_name, data_folder):
    if dataset_name in ['ENZYMES', 'COX2', 'PROTEINS', 'BZR', 'DD', 'NCI109']:
        dataset = TUDataset(f'{data_folder}/TUDataset/', name=dataset_name)
    elif dataset_name in ['IMDB-BINARY', 'REDDIT-BINARY', 'NCI1', 'Mutagenicity']:
        dataset = TUDataset(f'{data_folder}/TUDataset/', name=dataset_name)
    else:
        return None, -1, -1

    input_dim = dataset.num_features
    output_dim = dataset.num_classes
    print('{:22s}'.format(dataset_name),
          ' & {:5d}'.format(len(dataset)),
          ' & {:6.2f}'.format(dataset.data.x.shape[0] / len(dataset)),
          ' & {:6.2f}'.format(dataset.data.edge_index.shape[1] / len(dataset)),
          ' & {:5d}'.format(input_dim),
          ' & {:5d}'.format(output_dim),
          ' & Graph \\\\'
          )

    return dataset, input_dim, output_dim


def GraphDownstream(data, shots=5, test_fraction=0.4):
    num_classes = int(data.y.max().item()) + 1

    if data.name == 'bbbp':
        train_graph_list = []
        test_graph_list = []
        for c in range(num_classes):
            indices = torch.where(data.y.squeeze() == c)[0].tolist()
            for graph in empty_graphs:
                if graph in indices:
                    indices.remove(graph)
            random.shuffle(indices)
            train_graph_list.extend(indices[:shots])
            test_graph_list.extend(indices[shots:int(test_fraction * len(data) / 2)])

        train_data = [data[g] for g in train_graph_list]
        test_data = [data[g] for g in test_graph_list]

    else:
        graph_list = []
        for c in range(num_classes):
            indices = torch.where(data.y.squeeze() == c)[0].tolist()
            if len(indices) < shots:
                graph_list.extend(indices)
            else:
                graph_list.extend(random.sample(indices, k=shots))
        random_graph_list = random.sample(range(len(data)), k=len(data))
        for graph in graph_list:
            random_graph_list.remove(graph)
        train_data = [data[g] for g in graph_list]
        test_data = [data[g] for g in random_graph_list[:int(test_fraction * len(data))]]

    return train_data, test_data


if __name__ == '__main__':
    dataset_list = ['NCI109']
    for dataset_name in dataset_list:
        dataset, input_dim, output_dim = load_graph_data(dataset_name=dataset_name, data_folder='./data')
    train_data, test_data = GraphDownstream(dataset, shots=50)
    print(1)
