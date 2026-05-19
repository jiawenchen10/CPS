# Identifying representative subgraphs for reconstructing complex networks
This repository is designed to expose program code and network datasets from the article "Identifying representative subgraphs for reconstructing complex networks".

**Abstract**：
Inferring the structure of complex networks is essential for understanding natural and social systems. However, prevailing methods struggle to extract representative substructures, as reliance on the highest-centrality nodes or community-based sampling often fails to preserve structural features. 
Here, we adopt the generative learning framework with variational inference principle, which reconstructs network structures by integrating centrality measures with probabilistic sampling. 
In contrast to deterministic the highest-centrality node selection, centrality-probabilistic subgraphs preserve topological properties of the original networks, including degree distribution, heterogeneity and sparsity. With only 5\% to 20\% of nodes, centrality-probabilistic subgraphs achieve over 80\% reconstruction accuracy, as measured by link prediction tasks, and outperforms ten benchmark methods. 
Validated on real-world citation, collaboration, and biological networks, the probabilistic sampled subgraphs with degree, core, pagerank contribute most to network reconstruction, We further establish that subgraphs preserving degree distribution and sparsity of original network exhibit representational fidelity. This work advances the representative substructure identification solutions for network reconstruction, with applications in neuroscience, biology and sociology.


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
