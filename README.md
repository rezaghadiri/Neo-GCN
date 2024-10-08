
# Neo-GCN: Optimizing Identity-aware Graph Convolutional Networks with Natural Gradient Descent

Welcome to the **Neo-GCN** project! This repository contains the implementation of Neo-GCN, an innovative approach that integrates **Natural Gradient Descent (NGD)** with **Identity-aware Graph Neural Networks (ID-GNN)** to enhance the performance of Graph Convolutional Networks (GCNs). Neo-GCN addresses key challenges in GNNs such as slow convergence, gradient vanishing, and over-smoothing, resulting in improved accuracy and scalability in graph-based learning tasks.

## Key Features
- **Natural Gradient Descent (NGD)**: Efficient optimization that accounts for the geometry of the parameter space, enabling faster convergence and better generalization.
- **Identity-aware GCN**: Preserves node-specific identity features to prevent over-smoothing, enhancing the model's ability to differentiate nodes in deeper networks.
- **Benchmark Performance**: Outperforms traditional GCNs and other state-of-the-art models on widely-used datasets such as Cora, CiteSeer, and PubMed.

## Highlights
- **Natural Gradient Descent**: NGD uses the Fisher Information Matrix (FIM) for more accurate and efficient updates in GCN optimization, leading to more stable training.
- **Identity-aware Mechanisms**: Incorporates identity features into the GCN model to retain unique node characteristics, improving expressiveness and classification accuracy.
- **Improved Scalability**: By integrating NGD and identity-aware mechanisms, Neo-GCN scales effectively, even in deeper GNN architectures.

## Datasets
Neo-GCN has been evaluated on the following benchmark datasets:
- **Cora**: Citation network with 2,708 nodes and 5,429 edges.
- **CiteSeer**: Citation network with 3,327 nodes and 4,732 edges.
- **PubMed**: Citation network with 19,717 nodes and 44,338 edges.

## Results
Neo-GCN achieves state-of-the-art classification accuracy across these datasets, outperforming other GNN models. Some notable results include:
- **Cora**: 91.04% accuracy
- **CiteSeer**: 80.67% accuracy
- **PubMed**: 88.27% accuracy

## Methodology
Neo-GCN combines the following approaches:
1. **Identity-aware Convolution**: Preserves node-specific identity features in the graph convolution process to avoid over-smoothing.
2. **Natural Gradient Descent**: Uses NGD for GCN optimization by considering the non-Euclidean nature of graph data, allowing more efficient training.

For a detailed explanation of the methodology, please refer to the paper included in this repository.

## Citation
If you use Neo-GCN in your research, please cite:
