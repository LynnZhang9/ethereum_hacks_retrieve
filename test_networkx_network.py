import pickle
import networkx as nx
import matplotlib.pyplot as plt
def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


# G = load_pickle('./MulDiGraph.pkl')
data_path = r"C:\Users\linzhang2\Documents\Master Thesis\eth_dataset\Xblock\Transaction Dataset\Labeled dataset\Ethereum Phishing Transaction Network\archive\Ethereum Phishing Transaction Network\MulDiGraph.pkl"
G = load_pickle(data_path)
print(nx.info(G))
print('start drawing')

plt.subplot(111)
nx.draw(G)  # networkx draw()
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