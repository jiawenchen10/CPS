import networkx as nx
import pickle
from input_data import load_data


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



datasets =  ['ba3k_30','cora','citeseer', 'pubmed','cath','cmat','ws3k']
for dataset in datasets:
    adj, _ = load_data(dataset)

    G = nx.from_scipy_sparse_array(adj,create_using=nx.DiGraph())
    G.remove_edges_from(nx.selfloop_edges(G))
    r = 2
    
    file_path = "cevae/centrality/%s_degree.pkl" % dataset
    with open(file_path, "rb") as f:
        degree = pickle.load(f)
    print(f' dataset {dataset} done!')
    
    gravitydegree = gravity(G,degree,r)
    with open("cevae/centrality/%s_gravitydegree.pkl"%dataset, "wb") as f:
        pickle.dump(gravitydegree, f)
    
    file_path = "cevae/centrality/%s_core.pkl" % dataset
    with open(file_path, "rb") as f:
        core = pickle.load(f)

    gravitycore = gravity(G, core, r)
    with open("cevae/centrality/%s_gravitycore.pkl"%dataset, "wb") as f:
        pickle.dump(gravitycore, f)


    file_path = "cevae/centrality/%s_pagerank.pkl" % dataset
    with open(file_path, "rb") as f:
        pagerank = pickle.load(f)


    gravitypagerank = gravity(G,pagerank,r)
    with open("cevae/centrality/%s_gravitypagerank.pkl"%dataset, "wb") as f:
        pickle.dump(gravitypagerank , f)

    eigenvector = nx.eigenvector_centrality(G)
    with open("cevae/centrality/%s_eigenvector.pkl"%dataset, "wb") as f:
        pickle.dump(eigenvector, f)
    print(f' dataset {dataset} done!')

    
    betweenness = nx.betweenness_centrality(G)
    with open("cevae/centrality/%s_betweenness.pkl"%dataset, "wb") as f:
        pickle.dump(betweenness, f)


    gravitybetweenness = gravity(G, betweenness, r)
    with open("cevae/centrality/%s_gravitybetweenness.pkl"%dataset, "wb") as f:
        pickle.dump(gravitybetweenness, f)


    gravityeigenvector = gravity(G, eigenvector, r)
    with open("cevae/centrality/%s_gravityeigenvector.pkl"%dataset, "wb") as f:
        pickle.dump(gravityeigenvector, f)


    closeness = nx.closeness_centrality(G)
    with open("cevae/centrality/%s_closeness.pkl"%dataset, "wb") as f:
        pickle.dump(closeness, f)

    gravitycloseness = gravity(G, closeness, r)
    with open("cevae/centrality/%s_gravitycloseness.pkl"%dataset, "wb") as f:
        pickle.dump(gravitycloseness, f)

    print(f' dataset {dataset} done!')


