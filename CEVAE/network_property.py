
import numpy as np
import networkx as nx
import numpy as np
from scipy.stats import entropy

def fit_power_law(subgraph):
    degrees = [d for n, d in subgraph.degree()]
    
    degree_counts = np.bincount(degrees)
    # Remove the case where the degree is 0
    degree_counts = degree_counts[1:]   
    x = np.arange(1, len(degree_counts) + 1)
    y = degree_counts
    mask = y != 0
    x = x[mask]
    y = y[mask] 
    x_log = np.log(x)
    y_log = np.log(y)
    coeff = np.polyfit(x_log, y_log, 1) 
    y_coeff = -coeff[0]
    return y_coeff



def avg_shortest_path_length(G):
    try:
        # Check if the graph is a connected undirected graph
        if not G.is_directed():
            if nx.is_connected(G):
                average_shortest_path_length = nx.average_shortest_path_length(G)
            else:
                # The graph is not connected, calculate the average shortest path length of each connected component
                connected_components = list(nx.connected_components(G))
                if len(connected_components) > 1:
                    # print('Input graph is not connected; computing for each component:')
                    avg_spl_per_component = []
                    for component in connected_components:
                        subgraph = G.subgraph(component)
                        if nx.is_connected(subgraph):
                            avg_spl_per_component.append(nx.average_shortest_path_length(subgraph))
                    if avg_spl_per_component:
                        average_shortest_path_length = sum(avg_spl_per_component) / len(avg_spl_per_component)
                    else:
                        average_shortest_path_length = None
                else:
                    print("Graph is not connected.")
                    average_shortest_path_length = None
        else:
            raise ValueError("The input graph should be an undirected graph.")
    except nx.NetworkXError:
        average_shortest_path_length = None

    return  average_shortest_path_length 



def subgraph_property(G):

    density = nx.density(G)
    sparsity = 1 - density

    average_degree = sum(dict(G.degree()).values()) / len(G)


    average_clustering_coefficient = nx.average_clustering(G)

    # Average shortest path length
    average_shortest_path_length = avg_shortest_path_length(G)
    # Generate the corresponding random network (Erdős–Rényi model)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    G_ER = nx.gnm_random_graph(n, m)

    # Calculate the average clustering coefficient and average shortest path length of a random network
    C_ER = nx.average_clustering(G_ER)
    try:
        L_ER = avg_shortest_path_length(G_ER)
    except nx.NetworkXError:
        L_ER = None

    # Calculate r_C and r_L
    if average_clustering_coefficient != 0 and C_ER is not None:
        r_C = abs(average_clustering_coefficient - C_ER) / average_clustering_coefficient
    else:
        r_C = None

    if average_shortest_path_length is not None and L_ER is not None and average_shortest_path_length != 0:
        r_L = abs(average_shortest_path_length - L_ER) / average_shortest_path_length
    else:
        r_L = None

 
    return r_L,r_C, sparsity, average_degree, average_clustering_coefficient, average_shortest_path_length


def calculate_gini(graph):
    """
    Calculate the Gini coefficient of the degree distribution of the graph 
    (normalized, suitable for comparison of graphs of different sizes)
    """
    degrees = sorted(graph.degree(node) for node in graph.nodes())
    n = len(degrees)
    if n == 0 or sum(degrees) == 0:
        return 0.0
    
    # Calculate the Gini coefficient (the formula already implies scale invariance)
    sum_degrees = sum(degrees)
    sum_product = sum(i * degree for i, degree in enumerate(degrees, 1))
    gini = (2 * sum_product) / (n * sum_degrees) - (n + 1) / n


    return max(0.0, min(gini, 1.0))   

def calculate_normalized_entropy(graph):
    """
    Calculate normalized entropy (eliminate scale effects)
    """
    probabilities = get_degree_distribution(graph)
    filtered = probabilities[probabilities > 0]

    if len(filtered) <= 1:
        return 0.0
    
    entropy = -np.sum(filtered * np.log(filtered))
    max_entropy = np.log(len(filtered))  
    # The maximum entropy is log(number of different degrees)


    return entropy / max_entropy if max_entropy != 0 else 0.0


def align_and_smooth_distributions(P, Q, epsilon=1e-10, num_bins=50):
    # Alignment distribution length
    max_len = max(len(P), len(Q))
    min_len = min(len(P), len(Q))

    # If the distribution lengths are different, interpolate
    if len(P) != len(Q):
        x_old_P = np.linspace(0, 1, len(P))
        x_old_Q = np.linspace(0, 1, len(Q))
        x_new = np.linspace(0, 1, max_len)

        P_interpolated = np.interp(x_new, x_old_P, P)
        Q_interpolated = np.interp(x_new, x_old_Q, Q)
    else:
        P_interpolated = P
        Q_interpolated = Q

    # Smoothing
    P_smoothed = (P_interpolated + epsilon) / (1 + epsilon * max_len)
    Q_smoothed = (Q_interpolated + epsilon) / (1 + epsilon * max_len)

    return P_smoothed, Q_smoothed



def get_degree_distribution(graph):
    """ 
    Compute the degree distribution of the graph (including empty graph processing)
    """
    degrees = [graph.degree(node) for node in graph.nodes()]
    if not degrees:
        return np.ones(1) / 1  # Empty graph returns uniform distribution
    
    max_degree = max(degrees)
    hist = np.bincount(degrees, minlength=max_degree + 1)
    
    # Convert to probability distribution
    probabilities = hist / len(degrees)
    return probabilities



def align_distributions_with_binning(degrees_P, degrees_Q, num_bins=20, epsilon=1e-10):
    """
    Align the degree distributions of the two networks to eliminate the scale difference
    degrees_P and degrees_Q are lists of degree distributions of the two networks (each element represents the degree value of a node)
    """
    # Calculate the maximum degree of each of the two networks
    if len(degrees_P) == 0:
        max_degree_P = 0
    else:
        max_degree_P = max(degrees_P)
    
    if len(degrees_Q) == 0:
        max_degree_Q = 0
    else:
        max_degree_Q = max(degrees_Q)

    # Separate bins for each network
    def bin_degrees(degrees, max_degree, num_bins, epsilon):
        if max_degree == 0:
            return np.array([1.0])  # All degree values ​​are 0, and a bin with a probability of 1 is directly returned
        # Binning within the degree range of the network
        bins = np.linspace(0, max_degree, num=num_bins + 1)
        # Binning Statistics
        hist, _ = np.histogram(degrees, bins=bins)
        # Normalize and smooth
        hist = (hist + epsilon) / (hist.sum() + epsilon * num_bins)
        return hist

    # Binning the two networks separately
    hist_P = bin_degrees(degrees_P, max_degree_P, num_bins, epsilon)
    hist_Q = bin_degrees(degrees_Q, max_degree_Q, num_bins, epsilon)

    return hist_P, hist_Q




def calculate_kl_divergence(G1, G2):
    P = get_degree_distribution(G1)
    Q = get_degree_distribution(G2)
    # P, Q = align_and_smooth_distributions(P, Q)
    P, Q = align_distributions_with_binning(P,Q)
    # Ensure that the probability sums to 1 (to prevent numerical errors)
    P /= P.sum()
    Q /= Q.sum()

    kl = np.sum(P * np.log(P/Q))
    k2 = np.sum(Q * np.log(Q/P))
    normalized_kl = 1 - np.exp(-kl)
    normalized_k2 = 1 - np.exp(-k2)
    # kl_divergence = np.sum(P * np.log(P / Q))
    return normalized_kl, normalized_k2


def calculate_js_divergence(G1, G2):
    P = get_degree_distribution(G1)
    Q = get_degree_distribution(G2)
    # P, Q = align_and_smooth_distributions(P, Q)
    P, Q = align_distributions_with_binning(P,Q)
    M = 0.5 * (P + Q)
    js_divergence = 0.5 * np.sum(P * np.log(P / M)) + 0.5 * np.sum(Q * np.log(Q / M))
    return js_divergence


def calculate_gini(degrees):
    """Calculate the Gini coefficient of the degree sequence"""
    if len(degrees) == 0:
        return 0.0
    degrees = sorted(degrees)
    n = len(degrees)
    total = sum(degrees)
    if total == 0:
        return 0.0
    return (2 * sum(i * d for i, d in enumerate(degrees, 1))) / (n * total) - (n + 1) / n

def calculate_normalized_entropy(graph):
    """Calculate normalized entropy (eliminate scale effects)"""
    degrees = [graph.degree(node) for node in graph.nodes()]
    unique_vals, counts = np.unique(degrees, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(unique_vals)) if len(unique_vals) > 0 else 0
    return entropy / max_entropy if max_entropy > 0 else 0

def calculate_heterogeneity_diff(G1, G2):
    """
    Comparison of heterogeneity differences across scaled graphs (correct implementation)
    G1: original graph (large graph)
    G2: subgraph (small graph)
    """
    # 1. Direct comparison of Gini coefficients (scale invariance)
    gini_G1 = calculate_gini([G1.degree(node) for node in G1.nodes()])
    gini_G2 = calculate_gini([G2.degree(node) for node in G2.nodes()])
    gini_diff = abs(gini_G1 - gini_G2)
    
    # 2. Normalized entropy comparison
    entropy_diff = abs(calculate_normalized_entropy(G1) - calculate_normalized_entropy(G2))
    
    return gini_diff, entropy_diff




 
