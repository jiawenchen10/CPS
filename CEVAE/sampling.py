from preprocessing import sparse_to_tuple
import networkx as nx
import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf  
tf.disable_v2_behavior()
import warnings as wn
from input_data import load_data
import json
flags = tf.app.flags
FLAGS = flags.FLAGS
import pickle
from scipy.sparse import csr_matrix   
import operator
wn.simplefilter('ignore', UserWarning)

def sigmoid(x):
    return 1/(1+np.exp(-x))


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
    combined_centrality = {}
    max_centrailty1 = max(centrality1.values())
    max_centrality2 = max(centrality2.values())
    max_centrality3 = max(centrality3.values())
    for node in G.nodes():
        combined_centrality[node] = centrality1[node] / max_centrailty1+ \
                                  centrality2[node] / max_centrality2+ centrality3[node] / max_centrality3
    return combined_centrality


def hybrid_centrality(G,centrality1,centrality2):
    hybrid_centrality = {}
    max_centrality1 = max(centrality1.values())
    max_centrality2 = max(centrality2.values())
    for node in G.nodes():
        hybrid_centrality[node] = centrality1[node]/max_centrality1 *  \
                                  centrality2[node]/max_centrality2
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
    # Consider dividing by the largest Kshell value to normalize it
    return kshell

def get_distribution(measure,datasets, alpha, adj):
    """ Compute the p_i probabilities to pick each node i through the node sampling scheme of cevae  
    :param measure: node importance measure, among 'degree', 'core', 'uniform' and so on
    :param alpha: alpha scalar hyperparameter for degree and core sampling
    :param adj: sparse adjacency matrix of the graph
    :return: list of p_i probabilities of all nodes
    """
    r = 2  # Initial neighbor node radius
    measures = ['degree','core','pagerank','betweenness','eigenvector','closeness',
                'gravitydegree','gravitycore','gravitypagerank','gravitybetweenness','gravityeigenvector','gravitycloseness']
    if measure in measures:
        filename = 'E:/Ai-Learning-Practice/cndp/centrality/%s_%s.pkl' % (datasets, measure)
        with open(filename, "rb") as f:
            degree = pickle.load(f)
        proba = np.power(list(degree.values()),alpha)
    elif measure == 'uniform':
        # Uniform distribution
        proba = np.ones(adj.shape[0])
    else:
        raise ValueError('Undefined sampling method!')

    # Normalization
    proba = proba / np.sum(proba)
    return proba




def top_nodes_sampling(adj,datasets,measure,n):
    """The fuction used to delete unimportant edges to simplify the networks
    :param data: (Graph G)
    :param measure: centrality choice
    :param n: undeleted nodes
    :return: after simplify networks' adj
    """
    measures = ['degree','core','pagerank','betweenness','eigenvector','closeness',
                'gravitydegree','gravitycore','gravitypagerank','gravitybetweenness','gravityeigenvector','gravitycloseness']
    if measure in measures:
        filename = 'E:/Ai-Learning-Practice/cndp/centrality/%s_%s.pkl' % (datasets, measure)
        with open(filename, "rb") as f:
            centrality = pickle.load(f)
        sorted_centrality = sorted(centrality.items(),key = lambda x:x[1], reverse=True)

    else:
        raise ValueError('Undefined measure!')

    # Select the Top-n nodes of networks nodes and recording their connections
    top_nodes = [node[0] for node in sorted_centrality[:n]]
    # Sparse adjacency matrix of sampled subgraph
    sampled_adj = adj[top_nodes,:][:,top_nodes]
    sampled_adj_tuple = sparse_to_tuple(sampled_adj + sp.eye(sampled_adj.shape[0]))

    return top_nodes, sampled_adj_tuple, sampled_adj


def node_sparse_sampling(adj,datasets,measure,num_sampled_nodes,node_start):
    ''' To investigate the sparse contributes to the network reconstruction
    function input:
    :param adj: the adjacency  matrix of the original graph
    :param node_centrality: node centrality
    :param: num_sampled_nodes: the nodes' number of the subgraph
    :param: node_start: the initial node start [node_start, node_start + num_sampled_nodes]
    return:
    '''
    measures = ['degree','core','pagerank','betweenness','eigenvector','closeness',
                'gravitydegree','gravitycore','gravitypagerank','gravitybetweenness','gravityeigenvector','gravitycloseness']
    if measure in measures:
        filename = 'E:/Ai-Learning-Practice/cndp/centrality/%s_%s.pkl' % (datasets, measure)
        with open(filename, "rb") as f:
            centrality = pickle.load(f)
        sorted_key = sorted(centrality.items(),key = lambda x:x[1], reverse=True)

    else:
        raise ValueError('Undefined measure!')

    sampled_nodes = [node[0] for node in sorted_key[node_start:node_start+num_sampled_nodes]]
    # process this operation and get the different adj sparsity
    sampled_adj = adj[sampled_nodes,:][:,sampled_nodes]
    sampled_adj_tuple = sparse_to_tuple(sampled_adj+sp.eye(len(sampled_nodes)))
    sampled_nodes_centralities = [node[1] for node in sorted_key[node_start:node_start+num_sampled_nodes]]
    average_centrality = np.average(sampled_nodes_centralities)
    return sampled_nodes,  sampled_adj_tuple, sampled_adj,average_centrality


def get_actual_probabilities(adj, some_other_parameters=None):

    density = np.sum(adj,axis=1)/(adj.shape[0]-1)
    density = np.squeeze(density).reshape(adj.shape[0],-1)
    actual_probs = 1 - density
    # actual_probs = density

    return actual_probs

def node_sampling(adj, distribution, nb_node_samples, replace=False):
    """ Sample a subgraph from a given node-level distribution
    :param adj: sparse adjacency matrix of the graph
    :param distribution: p_i distribution, from get_distribution()
    :param nb_node_samples: size (nb of nodes) of the sampled subgraph
    :param replace: whether to sample nodes with replacement
    :return: nodes from the sampled subgraph, and subgraph adjacency matrix
    """
    # Sample nb_node_samples nodes, from the pre-computed distribution
    exp_distribution = distribution  #np.exp(distribution)
    
    # Normalize and distribition.sum() = 1
    normalized_distribution = exp_distribution / np.sum(exp_distribution)

    sampled_nodes = np.random.choice(adj.shape[0], size = nb_node_samples,
                                     replace = replace, p = normalized_distribution )
    # Sparse adjacency matrix of sampled subgraph
    sampled_adj = adj[sampled_nodes,:][:,sampled_nodes]
    # In tuple format (useful for optimizers)
    sampled_adj_tuple = sparse_to_tuple(sampled_adj + sp.eye(sampled_adj.shape[0]))
    return sampled_nodes, sampled_adj_tuple, sampled_adj


def node_uniform_sampling(adj, nb_node_samples, replace=False):
    """ Sample a subgraph uniformly
    :param adj: sparse adjacency matrix of the graph
    :param nb_node_samples: size (nb of nodes) of the sampled subgraph
    :param replace: whether to sample nodes with replacement
    :return: nodes from the sampled subgraph, and subgraph adjacency matrix
    """
    # Sample nb_node_samples nodes, from the pre-computed distribution


    sampled_nodes = np.random.choice(adj.shape[0], size = nb_node_samples,
                                     replace = replace)
    # Sparse adjacency matrix of sampled subgraph
    sampled_adj = adj[sampled_nodes,:][:,sampled_nodes]
    # In tuple format (useful for optimizers)
    sampled_adj_tuple = sparse_to_tuple(sampled_adj + sp.eye(sampled_adj.shape[0]))
    return sampled_nodes, sampled_adj_tuple, sampled_adj



def node_sampling_with_rejection(adj, distribution, nb_node_samples, replace=False):  
    """ Sample a subgraph from a given node-level distribution using Rejection Sampling  
      
    :param adj: sparse adjacency matrix of the graph  
    :param distribution: p_i distribution, from get_distribution() (proposed probabilities)  
    :param nb_node_samples: size (nb of nodes) of the sampled subgraph  
    :param replace: whether to sample nodes with replacement  
    :return: nodes from the sampled subgraph, and subgraph adjacency matrix  
    """  
    # Get the number of nodes
    num_nodes = adj.shape[0]  
    actual_probs = get_actual_probabilities(adj, some_other_parameters=None)  
    # Normalize both distributions so that they sum to 1  
    distribution = distribution / np.sum(distribution)  
    actual_probs = actual_probs / np.sum(actual_probs)  
      
    sampled_nodes = []  
    while len(sampled_nodes) < nb_node_samples:  
        # Sample a node index from the proposed distribution  
        proposed_index = np.random.choice(num_nodes, p=distribution)  
        # Calculate the acceptance probability  
        accept_prob = min(1, actual_probs[proposed_index] / distribution[proposed_index])   
        # Decide whether to accept or reject the sample  
        if np.random.rand() < accept_prob:  
            sampled_nodes.append(proposed_index)  
        # If sampling without replacement and the sample size is reached, break  
        if not replace and len(sampled_nodes) == nb_node_samples:  
            break  
      
    # Convert the list of nodes to a numpy array  
    sampled_nodes = np.array(sampled_nodes)  
      
    
    sampled_adj = adj[sampled_nodes,:][:,sampled_nodes]   # Sparse adjacency matrix of sampled subgraph  
      
    # Convert the sparse adjacency matrix to a tuple format (if needed)  
    # Assuming sparse_to_tuple is a function that converts a sparse matrix to a tuple  
    sampled_adj_tuple = sparse_to_tuple(sampled_adj + csr_matrix(np.eye(sampled_adj.shape[0])))  
      
    return sampled_nodes, sampled_adj_tuple, sampled_adj  


def node_sampling_gn(adj,distribution, nb_node_sampleds,num_layers = 5, replace=False):
    """ Sample a subgraph from a given node-level distribution using stratified sampling.
    :param adj: sparse adjacency matrix of the graph
    :param distribution: p_i distribution, from get_distribution()
    :param nb_node_samples: size (nb of nodes) of the sampled subgraph
    :param num_layers: number of layers for stratified sampling
    :param replace: whether to sample nodes with replacement (False)
    :return: nodes from the sampled subgraph, and subgraph adjacency matrix
    """
    # Number of nodes in each layer
    layer_size = len(distribution) //num_layers

    # Indices of nodes sorted by their probability in distribution
    sorted_indices = np.argsort(distribution)

    sampled_nodes = []

    for i in range(num_layers):
        start_idx = i * layer_size
        end_idx = (i + 1) * layer_size if i != num_layers-1 else len(distribution)

        layer_indices = sorted_indices[start_idx:end_idx]
        layer_distribution = distribution[layer_indices]
        #Normalize the distribution for the current layer
        layer_distribution /=layer_distribution.sum()  

        sampled_layer_nodes = np.random.choice(layer_indices, size = nb_node_sampleds//num_layers
                                               ,replace=replace, p = layer_distribution)
        
        sampled_nodes.extend(sampled_layer_nodes)   #(num_layers, nb_num_nodes/num_layers)
    
    sampled_nodes = np.array(sampled_nodes)

    if len(sampled_nodes) > nb_node_sampleds:
        sampled_nodes = np.random.choice(sampled_nodes, size=nb_node_sampleds, replace=False)


    sampled_adj = adj[sampled_nodes,:][:,sampled_nodes]


    #
    sampled_adj_tuple = sparse_to_tuple(sampled_adj + csr_matrix(np.eye(sampled_adj.shape[0])))  
      
    return sampled_nodes, sampled_adj_tuple, sampled_adj  


def mcmc_node_sampling(adj, distribution, nb_node_samples, burn_in=600):
    """ Sample a subgraph using MCMC from a given node-level distribution
    :param adj: sparse adjacency matrix of the graph
    :param distribution: p_i distribution, from get_distribution()
    :param nb_node_samples: number of nodes in the final sample
    :param burn_in: number of initial steps to discard
    :param mixing_time: number of steps between each sample to ensure independence
    :return: nodes from the sampled subgraph, and subgraph adjacency matrix
    """
    mixing_time=100
    # Initialize the chain with a random node
    current_node = np.random.randint(0, adj.shape[0])
    samples = [current_node]

    # Burn-in phase to reach the stationary distribution
    for _ in range(burn_in):
        next_node = np.random.choice(adj.shape[0], p=distribution)
        if adj[current_node, next_node] >=0:
            current_node = next_node

    # Sampling phase
    for _ in range(nb_node_samples - 1):
        for _ in range(mixing_time):
            next_node = np.random.choice(adj.shape[0], p=distribution)
            if adj[current_node, next_node] >= 0:     
                # Need to consider, if all are greater than 1, it is considered that all links exist, otherwise is there any value in this situation
                current_node = next_node
        if  len(np.unique(samples))<=nb_node_samples:
            samples.append(current_node)
        else:
            break
    # Convert list of sampled nodes into a unique set
    sampled_nodes = np.unique(samples)
    
    # Ensure we have enough samples
    while len(sampled_nodes) < nb_node_samples:
        next_node = np.random.choice(adj.shape[0], p=distribution)
        if adj[current_node, next_node] >=0:
            current_node = next_node
            sampled_nodes = np.append(sampled_nodes, current_node)
        sampled_nodes = np.unique(sampled_nodes)
    
    # Subgraph adjacency matrix
    sampled_adj = adj[sampled_nodes,:][:,sampled_nodes]

    # Convert to tuple format
    sampled_adj_tuple = sparse_to_tuple(sampled_adj + sp.eye(sampled_adj.shape[0]))

    return sampled_nodes, sampled_adj_tuple, sampled_adj




def gibbs_node_sampling(adj, nb_node_samples, burn_in=100, num_iterations=1000):
    """ Sample a subgraph using Gibbs sampling from a given graph
    :param adj: sparse adjacency matrix of the graph
    :param nb_node_samples: number of nodes in the final sample
    :param burn_in: number of initial iterations to discard
    :param num_iterations: total number of iterations for Gibbs sampling
    :return: nodes from the sampled subgraph, and subgraph adjacency matrix
    """
    
    # Convert the adjacency matrix to a CSR format for efficient row access
    adj_csr = csr_matrix(adj)
    
    # Initialize the state vector indicating whether a node is sampled or not
    state = np.zeros(adj.shape[0], dtype=bool)
    sampled_nodes = np.random.choice(adj.shape[0], size=nb_node_samples, replace=False)
    state[sampled_nodes] = True
    
    # Gibbs sampling loop
    for iteration in range(num_iterations + burn_in):
        # Shuffle the nodes to avoid any systematic bias
        shuffled_nodes = np.random.permutation(adj.shape[0])
        
        for node in shuffled_nodes:
            # Compute the probability of including the node given the others
            neighbors_included = adj_csr[node].multiply(state).sum()
            
            # Decide to include or exclude the node based on some criteria
            # For example, we can use a simple threshold based on the number of included neighbors
            if state[node]:
                # If the node is currently included, decide if it should be excluded
                if np.random.rand() < 1 / (neighbors_included + 1):  # Example criterion
                    state[node] = False
            else:
                # If the node is not included, decide if it should be included
                if np.random.rand() < neighbors_included / adj.shape[0]:  # Example criterion
                    state[node] = True
    
    # Extract the final sampled nodes
    sampled_nodes = np.where(state)[0]
    
    # Subgraph adjacency matrix
    sampled_adj = adj[sampled_nodes,:][:,sampled_nodes]
    
    # Convert to tuple format
    sampled_adj_tuple = sparse_to_tuple(sampled_adj + sp.eye(sampled_adj.shape[0]))
    
    return sampled_nodes, sampled_adj_tuple, sampled_adj