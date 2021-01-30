import pickle
import networkx as nx
import matplotlib.pyplot as plt
import math
def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


# G = load_pickle('./MulDiGraph.pkl')
data_path = r'./dataset/archive/EthereumPhishingTransactionNetwork/MulDiGraph.pkl'
# data_path = r"C:\Users\linzhang2\Documents\Master Thesis\eth_dataset\Xblock\Transaction Dataset\Labeled dataset\Ethereum Phishing Transaction Network\archive\Ethereum Phishing Transaction Network\MulDiGraph.pkl"
G = load_pickle(data_path)
print(nx.info(G))
print('start drawing')
res = []
for idx, nd in enumerate(nx.nodes(G)):
    if G.nodes[nd]['isp'] == 1:
        # print(nd)
        # print(G.nodes[nd]['isp'])
        res.append(nd)



subG = G.subgraph(res)
# pos = nx.spring_layout(subG, k=1, scale=5)
# pos = nx.planar_layout(subG, scale=5)
plt.subplot(111)
# nx.draw(subG, pos=pos, node_size=10)  # networkx draw()
nx.draw(subG, node_size=10)
plt.show()
print('done')

# 遍历结点：
# for idx, nd in enumerate(nx.nodes(G)):
#     print(nd)
#     print(G.nodes[nd]['isp'])
#     break

# # 遍历连边：
# for ind, edge in enumerate(nx.edges(G)):
#     (u, v) = edge
#     eg = G[u][v][0]
#     amo, tim = eg['amount'], eg['timestamp']
#     print(amo, tim)
#     break