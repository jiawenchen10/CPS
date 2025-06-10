import networkx as nx
import numpy as np
import operator
from  sampling import * 

def data_sampleprob(adj,measure,n):
    ''' Process the graph data to pick each node v by the centrality prob
    :param adj: the adjacenty matrix of snapshot graph
    :param measure: node importance, as degree, pagegrank, and so on
    :param n: the number of sampling nodes
    :return: the subgraph of sampling nodes with their edges
    '''
    alpha = 1 # alpha is flexible parameter to adjust the weight
    node_samples = n # sampling nodes number
    replace = False  # sampling nodes 'True or False', if Ture: the sampling strategy nodes raplace

    # sampling n nodes by centrality prob
    node_distribution = get_distribution(measure, alpha, adj)
    sampled_nodes, adj_label, adj_sampled_sparse = node_sampling(adj, node_distribution,
                                                                 node_samples, replace)
    #record the nodes with their edges of (u,v)
    edges_to_keep = []
    G = nx.from_numpy_matrix(adj,create_using=nx.DiGraph())
    for node in sampled_nodes:
        neighbors = list(G.neighbors(nodes))
        edges_to_keep.extend([(node, neighbor) for neighbor in neighbors])

    #delete the edges of nodes expect sampled nodes
    edges_to_remove = [(u,v) for u,v in G.edges() if u not in sampled_nodes or v not in sampled_nodes]
    G.remove_edges_from(edges_to_remove)
    adj_matrix = nx.adjacency_matrix(G)
    adj_matrix_dense = adj_matrix.toarray()

    return adj_matrix_dense

