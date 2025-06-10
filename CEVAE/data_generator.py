import networkx as nx
import numpy as np
import operator
from preprocessing import sparse_to_tuple
import scipy.sparse as sp
import pickle


def gravity(G,centrality, r):
    '''
    Paper: Identifying influential speaders by gravity models 2019   nature scientificreports
    Contribute: The paper proposed a novel method to caculate the mix-centrality by gravity theory
    - G： the networks of Snapshot graph
    - centrality: the type centrality of graph(generally list)
    - r: the radius of the neighbourhood of  centrality caculation to node v
    '''
    grav = {}
    for node in (G.nodes()):
        grav[node] = 0
        neighbour_nodes = list(G.neighbors(node))
        for neighbour in  neighbour_nodes:
            if (nx.shortest_path_length(G,source = node , target  = neighbour))>r:
                break
            if (node not in grav):
                grav[node] = 0
            if  (neighbour == node):
                break
            grav[node] += (centrality[neighbour] * centrality[neighbour])/((nx.shortest_path_length(G,source = node , target  = neighbour))**2)
            if ((nx.shortest_path_length(G,source = node , target  = neighbour)))<r:
                for n in G.neighbors(neighbour):
                    if (n not in neighbour_nodes):
                        neighbour_nodes.append(n)
        neighbour_nodes = []
    return grav


def combine_centrality(G, centrality1, centrality2, centrality3):
    '''
    Papers: Identifying influential spreaders in complex networks by an improved gravity model 2021
    and Identifying influential speaders by gravity model considering multi-characteristics of nodes 2022
    nature scientificreports
    contribute: use several centrality to improve the simple gravity methods to detect influencial nodes,
    especially to detect the k-shell(k-core) nodes' importance further
    '''
    combinecentraliy = {}
    max_centrailty1 = max(centrality1.values())
    max_centrality2 = max(centrality2.values())
    max_centrality3 = max(centrality3.values())
    for node in G.nodes():
        combinecentraliy[node] = centrality1[node] / max_centrailty1+ \
                                  centrality2[node] / max_centrality2+ centrality3[node] / max_centrality3
    return combinecentraliy


def hybrid_centrality(G,centrality1,centrality2):
    hybrid_centrality = {}
    max_centrality1 = max(centrality1.values())
    max_centrality2 = max(centrality2.values())
    if max_centrality1 == 0:
        max_centrality1 =1
    if max_centrality2 == 0:
        max_centrality2 =1
    for node in G.nodes():
        hybrid_centrality[node] = centrality1[node]/max_centrality1 * centrality2[node]/max_centrality2
    return hybrid_centrality


def h_index(G):
    ''' The function is different from the H_index of whole graph, and used to caculate nodes' h index value'''
    hindex = {}
    for node in G.nodes():
        max_hindex = G.degree(node)
        result = -1
        for i in range(0, max_hindex + 1):
            total_nodes = 0
            for neighbor in G.neighbors(node):
                if (G.degree(neighbor)>=i):
                    total_nodes = 1 + total_nodes
                if(total_nodes>=i):
                    result = max(result,i)
            hindex[node] = result
    return hindex

def extend_coreness(G):
    kshell = nx.core_number(G)
    coreness = {}
    for node in G.nodes():
        sum = 0
        for neighbor in G.neighbors(node):
            sum = sum + kshell[neighbor]
        coreness[node]  = sum
    extend_coreness = {}
    for node in G.nodes():
        sum = 0
        for neighbor in G.neighbors(node):
            sum = sum + coreness[neighbor]
        extend_coreness[node] = sum
    extend_coreness = {key: value/(nx.number_of_nodes(G)-1) for key, value in extend_coreness.items()}
    return extend_coreness

def imporved_kshell(G):
    '''
    The fuction is a special method to adjust the kshell in the gravity method
    Paper: Identifying influential speaders by gravity model considering multi-characteristics of nodes 2022
    nature scientificreports
    '''
    imporved_kshell = {}
    kshell = nx.core_number(G)
    kshell = sorted(kshell.items(),key = operator.itemgetter(1),reverse = True)
    kshell = {key:value/(nx.number_of_nodes(G)-1) for key, value in kshell.items()}

    return kshell


def processdata(adj,datasets,measure,n):
    """
    The fuction used to delete unimportant edges to simplify the networks
    :param data: (Graph G)
    :param measure: centrality choice
    :param n: undeleted nodes
    :return: after simplify networks' adj
    """
    # Calculate the centrality index of network nodes and rank them
    G = nx.from_scipy_sparse_matrix(adj)
    G.remove_edges_from(nx.selfloop_edges(G))


    r = 2  # Initial neighbor node radius
    if measure == 'degree':

        filename = 'centrality/%s_%s.pkl' % (datasets, measure)
        with open(filename, "rb") as f:
            degree = pickle.load(f)
        sorted_centrality = sorted(degree.items(),key = lambda x:x[1], reverse=True)

    elif measure == 'betweenness':

        filename = 'centrality/%s_%s.pkl' % (datasets, measure)
        with open(filename, "rb") as f:
            betweenness  = pickle.load(f)
        sorted_centrality = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)

    elif measure == 'eigenvector':

        filename = 'centrality/%s_%s.pkl' % (datasets, measure)
        with open(filename, "rb") as f:
            eigenvector  = pickle.load(f)
        sorted_centrality = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)

    elif measure == 'core':

        filename = 'centrality/%s_%s.pkl' % (datasets, measure)
        with open(filename, "rb") as f:
            coreness  = pickle.load(f)
        sorted_centrality = sorted(coreness.items(), key=lambda x: x[1], reverse=True)

    elif measure == 'pagerank':

        filename = 'centrality/%s_%s.pkl' % (datasets, measure)
        with open(filename, "rb") as f:
            pagerank  = pickle.load(f)
        sorted_centrality = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

    elif measure == 'closeness':

        filename = 'centrality/%s_%s.pkl' % (datasets, measure)
        with open(filename, "rb") as f:
            closeness  = pickle.load(f)

        sorted_centrality = sorted(closeness.items(), key=lambda x: x[1], reverse=True)


    elif measure == 'hindex':
        hindex = h_index(G)
        sorted_centrality = sorted(hindex.items(), key=lambda x: x[1], reverse=True)

    elif measure == 'gravitydegree':

        filename = 'centrality/%s_%s.pkl' % (datasets, measure)
        with open(filename, "rb") as f:
            centrality  = pickle.load(f)

        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    elif measure == 'gravitycore':

        filename = 'centrality/%s_%s.pkl' % (datasets, measure)
        with open(filename, "rb") as f:
            centrality  = pickle.load(f)

        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    elif measure == 'gravitypagerank':

        filename = 'centrality/%s_%s.pkl' % (datasets, measure)
        with open(filename, "rb") as f:
            centrality  = pickle.load(f)
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)


    elif measure == "gravitybetweenness":

        filename = 'centrality/%s_%s.pkl' % (datasets, measure)
        with open(filename, "rb") as f:
            centrality  = pickle.load(f)
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)


    elif measure == 'gravityeigenvector':

        filename = 'centrality/%s_%s.pkl' % (datasets, measure)
        with open(filename, "rb") as f:
            centrality  = pickle.load(f)
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)


    elif measure == 'gravitycloseness':

        filename = 'centrality/%s_%s.pkl' % (datasets, measure)
        with open(filename, "rb") as f:
            centrality  = pickle.load(f)
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)


    elif measure == 'gravityhindex':
        centrality = gravity(G, hindex, r)
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    elif measure == 'coreness_eigen':
        centrality = hybrid_centrality(G, coreness, eigenvector)
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)


    elif measure == 'coreness_pagerank':
        centrality = hybrid_centrality(G, coreness, pagerank)
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    elif measure == 'coreness_hindex':
        centrality = hybrid_centrality(G, coreness, hindex)
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)


    elif measure == 'coreness_betweenness':
        betweenness = nx.betweenness_centrality(G)
        centrality = hybrid_centrality(G, coreness, betweenness)
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)



    elif measure == 'uniform':
        centrality = {node: 1/nx.number_of_nodes(G) for node in G.nodes()}
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    else:
        raise ValueError('Undefined measure!')

    # Select the top n nodes in terms of centrality and record their edges
    top_nodes = [i for i,_ in sorted_centrality[:n]]
    sampled_nodes = top_nodes

    #Sparse adjacency matrix of sampled subgraph
    sampled_adj = adj[sampled_nodes,:][:,sampled_nodes]
    sampled_adj_tuple = sparse_to_tuple(sampled_adj + sp.eye(sampled_adj.shape[0]))

    return sampled_nodes, sampled_adj_tuple, sampled_adj









