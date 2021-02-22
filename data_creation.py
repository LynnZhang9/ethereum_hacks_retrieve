from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx

import pickle
import networkx as nx



class Ethereum_Dataset(InMemoryDataset):
    def __init__(self, datalist) -> None:
        super().__init__()
        self.data, self.slices = self.collate(datalist)



def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def get_data(data_path):
    G = load_pickle(data_path)
    print(nx.info(G))
    res_1 = []
    for idx, nd in enumerate(nx.nodes(G)):
        if G.nodes[nd]['isp'] == 1:
            # print(nd)
            # print(G.nodes[nd]['isp'])
            res_1.append(nd)
        if len(res_1) >= 10:
            break
    res_0 = []
    for idx, nd in enumerate(nx.nodes(G)):
        if G.nodes[nd]['isp'] == 0:
            # print(nd)
            # print(G.nodes[nd]['isp'])
            res_0.append(nd)
        if len(res_0) >= 10:
            break
    res = res_0 + res_1
    subG = G.subgraph(res)

    data = from_networkx(subG)
    print('converting is finished')
    return data



if __name__ == '__main__':

    data_path = r'./home/lin/workspace/ethereum_hacks_retrieve/dataset/archive/EthereumPhishingTransactionNetwork/MulDiGraph.pkl'
    data = get_data(data_path)
    data_list = [data]
    myData = Ethereum_Dataset(data_list)
    print('DONE')
    # data_path = r"C:\Users\linzhang2\Documents\Master Thesis\eth_dataset\Xblock\Transaction Dataset\Labeled dataset\Ethereum Phishing Transaction Network\archive\Ethereum Phishing Transaction Network\MulDiGraph.pkl"
