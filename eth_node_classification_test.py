import sys
sys.path.append('../')
from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier
from autogl.module import Acc
import yaml
import random
import torch
import numpy as np
import os
import logging
logging.basicConfig(level=logging.INFO)


from torch_geometric.data import InMemoryDataset
# from torch_geometric.utils.convert import from_networkx
import torch_geometric.data
import pickle
import networkx as nx




class Ethereum_Dataset(InMemoryDataset):
    def __init__(self, datalist) -> None:
        super().__init__()
        self.data, self.slices = self.collate(datalist)

def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def from_networkx(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """

    G = nx.convert_node_labels_to_integers(G, label_attribute='address')
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]
    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass
    # data['x'] = torch.tensor(list(range(0, len(G.nodes))))
    data['x'] = torch.zeros([20, 1], dtype=torch.int32)
    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data = torch_geometric.data.Data(x=data.x, amount=data.amount,edge_index=data.edge_index, y=data.isp, timestamp=data.timestamp, address=data.address)
    data.num_nodes = G.number_of_nodes()
    data['train_mask'] = torch.tensor([True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False])
    data['val_mask'] = torch.tensor([False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, False, False, False, False, False])
    data['test_mask'] = torch.tensor([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True])
    # data = data.random_splits_mask(train_ratio=0.5, val_ratio=0.3)

    return data

def get_data(data_path):
    G = load_pickle(data_path)
    print(nx.info(G))
    # res_1 = []
    # for idx, nd in enumerate(nx.nodes(G)):
    #     if G.nodes[nd]['isp'] == 1:
    #         # print(nd)
    #         # print(G.nodes[nd]['isp'])
    #         res_1.append(nd)
    #     if len(res_1) >= 10:
    #         break
    # res_0 = []
    # for idx, nd in enumerate(nx.nodes(G)):
    #     if G.nodes[nd]['isp'] == 0:
    #         # print(nd)
    #         # print(G.nodes[nd]['isp'])
    #         res_0.append(nd)
    #     if len(res_0) >= 10:
    #         break
    # res = res_0 + res_1
    # subG = G.subgraph(res)
    # subgraph = nx.Graph(subG)
    # nx.write_gpickle(subgraph, "test_subgraph.gpickle")

    data = from_networkx(G)

    print('converting is finished')
    return data

if __name__ == '__main__':
    # data_path = r'/home/lin/workspace/ethereum_hacks_retrieve/dataset/archive/EthereumPhishingTransactionNetwork/MulDiGraph.pkl'
    data_path = r'test_subgraph.gpickle'
    data = get_data(data_path)
    data_list = [data]
    # myData = Ethereum_Dataset(data_list)

    from argparse import ArgumentParser
    parser = ArgumentParser()
    # parser.add_argument('--dataset', default='cora', type=str)
    parser.add_argument('--configs', type=str, default='../configs/nodeclf_gcn_benchmark_small.yml')
    # following arguments will override parameters in the config file
    parser.add_argument('--hpo', type=str, default='random')
    parser.add_argument('--max_eval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=0, type=int)
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    seed = args.seed
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # dataset = build_dataset_from_name(args.dataset)
    dataset = Ethereum_Dataset(data_list)
    
    configs = yaml.load(open(args.configs, 'r').read(), Loader=yaml.FullLoader)
    configs['hpo']['name'] = args.hpo
    configs['hpo']['max_evals'] = args.max_eval
    autoClassifier = AutoNodeClassifier.from_config(configs)

    # train
    # if args.dataset in ['cora', 'citeseer', 'pubmed']:
    #     autoClassifier.fit(dataset, time_limit=3600, evaluation_method=[Acc])
    # else:
    autoClassifier.fit(dataset, time_limit=3600, evaluation_method=[Acc], seed=seed, train_split=0.5, val_split=0.3, balanced=False)
    val = autoClassifier.get_model_by_performance(0)[0].get_valid_score()[0]
    print('val acc: ', val)

    # test
    predict_result = autoClassifier.predict_proba(use_best=True, use_ensemble=False)
    print('test acc: ', Acc.evaluate(predict_result, dataset.data.y[dataset.data.test_mask].numpy()))



