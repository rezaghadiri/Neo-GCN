
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

## Installation
To run Neo-GCN, clone the repository and install the required dependencies:
\`\`\`bash
git clone https://github.com/rezaghadiri/Neo-GCN
cd Neo-GCN
pip install -r requirements.txt
\`\`\`

## Usage
You can train the Neo-GCN model on any of the supported datasets:
\`\`\`bash
python train.py --dataset cora
\`\`\`

Available options include:
- \`--dataset\`: Choose between \`cora\`, \`citeseer\`, or \`pubmed\`.
- \`--optimizer\`: Specify the optimizer (\`adam\` or \`sgd\`).
- \`--learning_rate\`: Set the learning rate (default is \`0.01\`).
- \`--epochs\`: Number of training epochs (default is \`100\`).

## Methodology
Neo-GCN combines the following approaches:
1. **Identity-aware Convolution**: Preserves node-specific identity features in the graph convolution process to avoid over-smoothing.
2. **Natural Gradient Descent**: Uses NGD for GCN optimization by considering the non-Euclidean nature of graph data, allowing more efficient training.

For a detailed explanation of the methodology, please refer to the paper included in this repository.

## Citation
If you use Neo-GCN in your research, please cite:
\`\`\`bibtex
@article{ghadiri2024neo,
  title={A Neo-Approach for Node Classification: Optimizing Identity-aware Graph Convolutional Networks with Natural Gradient Descent in Citation Graphs},
  author={Reza Ghadiri, Mojtaba Ghadiri, and Soheila Ashkezari-Toussi},
  journal={Journal of LaTeX Class Files},
  year={2024}
}
\`\`\`

## License
This project is licensed under the MIT License.
