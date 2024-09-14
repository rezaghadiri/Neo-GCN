
# Neo-GCN: Optimizing Identity-Aware Graph Convolutional Networks with Natural Gradient Descent

![GitHub repo size](https://img.shields.io/github/repo-size/username/Neo-GCN) ![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen) ![License](https://img.shields.io/github/license/username/Neo-GCN)

## Overview

**Neo-GCN** is a cutting-edge approach for enhancing the performance of Graph Convolutional Networks (GCNs) by integrating Natural Gradient Descent (NGD) with Identity-aware Graph Neural Networks (ID-GCNs). Neo-GCN addresses key challenges in GCN optimization, such as slow convergence, over-smoothing, and gradient vanishing. By incorporating NGD, Neo-GCN efficiently optimizes weight parameters while preserving unique node characteristics through identity features. This results in improved classification accuracy, better generalization, and enhanced stability, even in deeper networks.

### Key Features
- **Natural Gradient Descent (NGD):** A powerful optimization method that leverages the geometry of the parameter space for faster convergence and more stable training.
- **Identity-aware Mechanism:** Incorporates unique node identity features into the graph convolution process, mitigating over-smoothing and boosting expressiveness.
- **Benchmark Performance:** Neo-GCN outperforms traditional GCNs and state-of-the-art variants in tasks like node classification on datasets such as Cora, CiteSeer, and PubMed.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Benchmarks](#benchmarks)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Installation

To get started with Neo-GCN, clone the repository and install the necessary dependencies.

```bash
git clone https://github.com/username/Neo-GCN.git
cd Neo-GCN
pip install -r requirements.txt
```

Make sure you have PyTorch, PyG, and other relevant packages installed.

---

## Usage

### Training the Model
To train Neo-GCN on a specific dataset (e.g., Cora), use the following command:

```bash
python train.py --dataset cora --epochs 100 --learning_rate 0.01
```

The script supports various datasets, including **Cora**, **CiteSeer**, and **PubMed**. You can also fine-tune other hyperparameters, such as the learning rate, number of layers, and optimizer type.

### Arguments:
- `--dataset`: Choose between `cora`, `citeseer`, `pubmed`
- `--epochs`: Set the number of training epochs
- `--learning_rate`: Define the learning rate
- `--optimizer`: Choose between `adam`, `sgd`
  
---

## Benchmarks

Neo-GCN has been extensively tested and shows strong performance across multiple benchmark datasets. Below is a summary of its performance in node classification tasks.

| Dataset   | Accuracy (Neo-GCN) | Accuracy (GCN) | Accuracy (ID-GCN) |
|-----------|--------------------|----------------|-------------------|
| **Cora**  | 91.04%             | 89.44%         | 86.30%            |
| **CiteSeer** | 80.67%          | 79.80%         | 71.90%            |
| **PubMed**  | 88.27%           | 87.16%         | -                 |

Neo-GCN demonstrates its superiority, particularly in handling deeper networks with complex graph structures.

---

## Experimental Setup

We used the following benchmark datasets for training and evaluation:

- **Cora**: 2,708 nodes, 5,429 edges, 7 classes
- **CiteSeer**: 3,327 nodes, 4,732 edges, 6 classes
- **PubMed**: 19,717 nodes, 44,338 edges, 3 classes

For each dataset, we trained Neo-GCN using both SGD and Adam optimizers, with learning rates and other hyperparameters fine-tuned for optimal performance.

---

## Results

Neo-GCN consistently outperforms traditional GCN and ID-GCN models in classification tasks, particularly in datasets with complex node relationships. By integrating NGD and identity-aware mechanisms, Neo-GCN improves both the accuracy and stability of GCNs.

### Sample Results (Cora Dataset):
- **Test Accuracy**: 91.04%
- **Validation Accuracy**: 90.37%
- **Loss**: 0.45

---

## Contributing

We welcome contributions to Neo-GCN! To contribute, follow these steps:
1. Fork the repository
2. Create a new branch: `git checkout -b feature-branch`
3. Make your changes and commit: `git commit -m 'Add new feature'`
4. Push to the branch: `git push origin feature-branch`
5. Submit a pull request

Please ensure that your code adheres to the repository's code standards and includes appropriate tests.

---

## Citation

If you find Neo-GCN helpful in your research, please cite our paper as follows:

```
@article{ghadiri2024neo,
  title={A Neo-Approach for Node Classification: Optimizing Identity-Aware Graph Convolutional Networks with Natural Gradient Descent in Citation Graphs},
  author={Reza Ghadiri, Mojtaba Ghadiri, and Soheila Ashkezari-Toussi},
  journal={Journal of LaTeX Class Files},
  year={2024},
  volume={14},
  number={8},
  pages={1-12},
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

This work builds upon foundational research in the field of Graph Neural Networks and advanced optimization techniques. Special thanks to the research community for their invaluable contributions to the development of GNNs.

For more information, check out our full [paper](https://linktopaper.com) published in the Journal of LaTeX Class Files.

