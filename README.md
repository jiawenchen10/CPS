# Topology-Preserving Network Reconstruction via Centrality Based Subgraph Sampling
This repository is designed to expose program code and network datasets from the article "Topology-Preserving Network Reconstruction via Centrality Based Subgraph Sampling".

**Abstract**：
Inferring the structure of complex networks is essential for understanding natural and social systems. However, prevailing methods struggle to extract representative substructures, as reliance on high-centrality nodes or community-based sampling often fails to preserve global topological features. Here, we propose the Centrality-Enhanced Variational Autoencoder—a generative learning framework that robustly reconstructs network structures by integrating centrality measures with probabilistic sampling. \textcolor{red}{Unlike conventional paradigm of deterministic high-centrality node selection, our probabilistic centrality-based sampling strategy yields reconstructed networks whose topological properties, including degree distribution, heterogeneity and sparsity closely match those of the original graphs. Using only 5\% to 20\% of nodes, our method achieves over 80\% reconstruction accuracy, as measured by AUC and AP scores of link prediction, and consistently outperforms ten benchmark methods.} Evaluated across nine networks (seven empirical and two synthetic), our method identifies probabilistic subgraphs derived from attributes such as degree, coreness, and PageRank, which more effectively capture structural invariants. We further establish that subgraphs preserving both degree distribution and sparsity of original network exhibit optimal representational fidelity. This work advances scalable solutions for network reconstruction, with direct applications in real-world network analysis.


## Requirements
In order to be able to run the code, you need to install the packages contained in `requirements.txt`. We suggest to create a conda environment with
`conda create --name CEVAE --no-default-packages`, activate it with `conda activate CEVAE`, and install all the dependencies by running (inside `CEVAE` directory):

```bash
pip install -r requirements.txt
```


## Usage
To test the program on the given example scripts on Cora, sh scripts/cora.sh :  

```bash
for measures in pagerank core degree betweenness closeness gravitydegree gravitycore gravitypagerank gravitybetweenness gravitycloseness
  do
for method in sn tn rn un  
do 
    for numbers in $(seq 30 30  2708)
    do
        echo $measures
        python train.py --model=gcn_vae --dataset=cora  --task=link_prediction --fastgae=$method  --measure=$measures --alpha=1.0  --nb_node_samples=$numbers --learning_rate=0.008
    done
  done
done

```

You can find this list by running (inside `code` directory): 

```bash
python main.py --help
```
