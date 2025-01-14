import random
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import Planetoid, KarateClub, Flickr, Actor, Amazon
from torch_geometric.utils import k_hop_subgraph, structured_negative_sampling


def get_subgraphs(data, node_list, num_hops=2):
    graph_list = []
    for node in node_list:
        subset, edge_index, mapping, _ = k_hop_subgraph(node_idx=node,
                                                        num_hops=num_hops,
                                                        edge_index=data.edge_index,
                                                        relabel_nodes=True)
        subgraph_data = Data(x=data.x[subset], edge_index=edge_index, y=data.y[node], target_node=mapping.item())
        graph_list.append(subgraph_data)
    return graph_list


def load_node_data(dataset_name, data_folder):
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=f'{data_folder}/Planetoid', name=dataset_name)
    elif dataset_name == 'Flickr':
        dataset = Flickr(root=f'{data_folder}/Flickr')
    elif dataset_name in ['computers', 'photo']:
        dataset = Amazon(root=f'{data_folder}/Amazon', name=dataset_name)
    else:
        return None, -1, -1

    data = dataset[0]
    input_dim = dataset.num_features
    output_dim = dataset.num_classes

    return data, input_dim, output_dim


def NodePretrain(data, batch_size: int, num_hops=2):
    # split_ratio = 0.1
    # node_list = random.sample(range(data.num_nodes), k=int(split_ratio * data.num_nodes))
    iterations = 80
    assert batch_size * iterations < data.x.shape[0], 'Too many raining nodes'
    node_list = random.sample(range(data.num_nodes), k=batch_size * iterations)

    graph_list = get_subgraphs(data, node_list, num_hops)

    if len(graph_list) % batch_size > 0:
        graph_list = graph_list[:- (len(graph_list) % batch_size)]  # ensure the same num of graphs in each mini-batch

    return graph_list


def NodeDownstream(data, shots=5, test_fraction=0.2):
    num_classes = data.y.max().item() + 1
    node_list = []
    for c in range(num_classes):
        indices = torch.where(data.y.squeeze() == c)[0].tolist()
        if len(indices) < shots:
            node_list.extend(indices)
        else:
            node_list.extend(random.sample(indices, k=shots))
    random_node_list = random.sample(range(data.num_nodes), k=data.num_nodes)
    for node in node_list:
        random_node_list.remove(node)
    train_data = get_subgraphs(data, node_list)
    test_data = get_subgraphs(data, random_node_list[:int(test_fraction * data.num_nodes)])

    return train_data, test_data


if __name__ == '__main__':
    # dataname = 'Flickr'
    # data_folder = './data'
    # data, input_dim, output_dim = load_node_data(dataname, data_folder)
    data = KarateClub()[0]
    graph_list = NodePretrain(data, batch_size=3, num_hops=1)

    print(1)
