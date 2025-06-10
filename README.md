# Topology-Preserving Network Reconstruction via Centrality Based Subgraph Sampling
This repository is designed to expose program code and network datasets from the article "Topology-Preserving Network Reconstruction via Centrality Based Subgraph Sampling".

**Abstract**：
Inferring the structure of complex networks is essential for understanding natural and social systems. However, prevailing methods struggle to extract representative substructures, as reliance on high-centrality nodes or community-based sampling often fails to preserve global topological features. Here, we propose the Centrality-Enhanced Variational Autoencoder—a generative learning framework that robustly reconstructs network structures by integrating centrality measures with probabilistic sampling. \textcolor{red}{Unlike conventional paradigm of deterministic high-centrality node selection, our probabilistic centrality-based sampling strategy yields reconstructed networks whose topological properties, including degree distribution, heterogeneity and sparsity closely match those of the original graphs. Using only 5\% to 20\% of nodes, our method achieves over 80\% reconstruction accuracy, as measured by AUC and AP scores of link prediction, and consistently outperforms ten benchmark methods.} Evaluated across nine networks (seven empirical and two synthetic), our method identifies probabilistic subgraphs derived from attributes such as degree, coreness, and PageRank, which more effectively capture structural invariants. We further establish that subgraphs preserving both degree distribution and sparsity of original network exhibit optimal representational fidelity. This work advances scalable solutions for network reconstruction, with direct applications in real-world network analysis.


# Baseline model 
The replication code of the following paper work is used in the baseline model approach, as follows:



### GAE and VGAE Model.





### Graphite-GAE and Graphite-VAGE Model.





