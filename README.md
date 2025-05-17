# neoTextGCN: A Neo-Approach for Node Classification in Text-Attributed Graphs

This repository contains the implementation for the paper "neoTextGCN: A Neo-Approach for Node Classification in Text-Attributed Graphs".

## Abstract

Node classification in text-attributed graphs (TAGs) is a critical task in graph machine learning, with applications ranging from citation networks to social media analysis. Traditional approaches often rely on shallow text embeddings and complex graph neural network (GNN) architectures, leading to suboptimal performance and high computational costs. This paper introduces a novel approach. Initially, supervised parameter-efficient fine-tuning (PEFT) is conducted on a pre-trained language model (LM) tailored for the downstream task, such as node classification. Next, node embeddings are created from the last hidden states of the fine-tuned LM. These generated features can then be employed by an Identity-aware Graph Convolutional Network (ID-GCN) with Natural Gradient Descent (NGD) to optimize weight parameters efficiently by considering the geometry of the parameter space for training on the identical task. The approach is evaluated on the Cora, CiteSeer, and PubMed datasets, demonstrating significant improvements in classification accuracy while maintaining computational efficiency. The results highlight the power of integrating advanced language models with lightweight fine-tuning and identity-aware graph structures, offering a scalable and effective solution for textual graph learning.

## Overview

The proposed methodology, neoTextGCN, integrates advanced language model embeddings with an identity-aware graph neural network for node classification in text-attributed graphs. The approach consists of two main stages:

1.  **Text Embedding Generation**: Utilizes DeBERTa-v3-large, a transformer-based model, to generate contextual embeddings for node textual attributes (titles and abstracts). This model is fine-tuned using Low-Rank Adaptation (LoRA) on the specific node classification task.
2.  **Graph-Based Classification**: Employs an Identity-aware Graph Convolutional Network (ID-GCN) with Natural Gradient Descent (NGD) to perform node classification using the generated embeddings and the graph structure.

This approach aims to address limitations of prior methods such as shallow embeddings, complex joint training, and computational overhead.

## Key Features

* **Advanced Language Model Integration**: Leverages DeBERTa-v3-large for high-quality text embeddings.
* **Parameter-Efficient Fine-Tuning**: Employs LoRA for efficient adaptation of the large language model.
* **Identity-Aware Graph Neural Network**: Uses ID-GCN to preserve node-specific identity features and mitigate over-smoothing.
* **Efficient Optimization**: Incorporates Natural Gradient Descent (NGD) for optimizing GCN weight parameters by considering the geometry of the parameter space.
* **Improved Performance**: Demonstrates significant improvements in classification accuracy on benchmark datasets.

## Datasets

The approach was evaluated on three widely-used benchmark datasets:
* **Cora**: A citation network of 2,708 scientific papers with 5,429 citation links, classified into 7 research topics.
* **CiteSeer**: A citation network consisting of 3,312 publications with 4,732 citation links, categorized into 6 classes.
* **PubMed**: A dataset of 19,717 scientific publications from the PubMed database concerning diabetes, with 44,338 citation links, classified into 3 categories.

## Experimental Results

The neoTextGCN approach was compared against several baseline models including GCN, GraphSAGE, GAT, GIN, SplineCNN, and recent state-of-the-art methods like SimTeG and GRAD.

* On the **Cora** dataset, neoTextGCN (DeBERTaV3 with LoRA + Neo-GCN) achieved a test accuracy of 84.10 ± 2.30. Neo-GCN (ID-GCN with NGD) using SGD-KFAC achieved an accuracy of 91.03 ± 0.60.
* On the **CiteSeer** dataset, neoTextGCN (DeBERTaV3 with LoRA + Neo-GCN) achieved a test accuracy of 70.43 ± 1.15. Neo-GCN (ID-GCN with NGD) using Adam-KFAC achieved an accuracy of 80.67 ± 0.98.
* For the **PubMed** dataset, results for the full neoTextGCN approach faced scalability challenges due to hardware limitations. However, GCN (SGD-KFAC) achieved 89.36 ± 0.57, and SplineCNN reached 88.88 ± 0.0.

The results validate the effectiveness of combining the simplicity of approaches like SimTeG with advanced components like DeBERTa-v3-large, LoRA, and ID-GCN with NGD.

## Repository

The implementation of the Neo-GCN component is publicly available at:
[https://www.github.com/rezaghadiri/Neo-GCN](https://www.github.com/rezaghadiri/Neo-GCN)


## How to Cite

If you use this work, please cite the original paper:

```bibtex
@article{Ghadiri2025neoTextGCN,
  title={neoTextGCN: A Neo-Approach for Node Classification in Text-Attributed Graphs},
  author={Reza Ghadiri and Mansoor Fateh and Hoda Mashayekhi},
  journal={ArXiv preprint},
  year={2025},
  eprint={arXiv:xxxx.xxxxx} -- Placeholder, replace with actual arXiv ID if available
}
