import argparse


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--task', default='EdgePredGraphPrompt', type=str)
    parser.add_argument('--dataset_name', type=str, default='PubMed', help='Choose the dataset')
    parser.add_argument('--device', type=int, default=2, help='Which gpu to use if any (default: 0)')
    parser.add_argument('--gnn_type', type=str, default="GCN", help='GCN or GIN')
    parser.add_argument('--prompt_type', type=str, default='GraphPrompt',
                        help='Choose the prompt type for node or graph task, for node task,we support \GPPT\, \All-in-one\, \Gprompt\ for graph task , \All-in-one\, \Gprompt\, \GPF\, \GPF-plus\ ')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hideen layer of GNN dimensions (default: 128)')
    parser.add_argument('--batch_size', type=int, default=128, help='Input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train (default: 50)')
    parser.add_argument('--shots', type=int, default=20, help='Number of shots')
    parser.add_argument('--pre_train_model_path', type=str, default='None', help='add pre_train_model_path to the downstream task, the model is self-supervise model if the path is None and prompttype is None.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0, help='Weight decay (default: 0)')
    parser.add_argument('--GNN_layers', type=int, default=2, help='Number of GNN message passing layers (default: 2).')

    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='Dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean", help='Graph pooling (sum, mean, max)')
    parser.add_argument('--JK', type=str, default="last", help='How the node features across layers are combined. last, sum, max or concat')

    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--runseed', type=int, default=0, help="Seed for running experiments.")
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataset loading')
    parser.add_argument('--MLP_layers', type=int, default=1, help='A range of [1,2,3]-layer MLPs with equal width')
    parser.add_argument('--pnum', type=int, default=5, help='The number of independent basis for GPF-plus')

    args = parser.parse_args()

    return args
