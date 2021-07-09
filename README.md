# MP-GCN
H. Zhao, J. Xie, H. Wang, "Graph Convolutional Network based on Multi-head Pooling for Short Text Classification",2021.

## Overview
Here we provide the implementation of a MP-GCN in TensorFlow, along with a minimal execution example (on the MR dataset). The repository is organised as follows:

* data/  # contains the necessary dataset files for MR (single-graph data)
* data_m/  # contains multi-graph data files for MR
* vector/  # is used to store graph embedding
* build_multi_graph.py  # is used to create single graph
* build_single_graph.py  # is used to create multiple graphs
* inits.py  # contains parameter initialization functions
* layers.py  # contains the calculation function of single layer graph convolution
* metrics.py  # contains evaluation functions
* models.py/  # contains the implementation of the MP-GCN and GCN
* train.py  # contains the main function
* utils.py  # contains common tool functions
* visualize.py  # visualization of document graph embedding
* visualize_words.py  # visualization of word graph embedding

## Datesets
We ran our experiments on five widely used benchmark corpora including 20-Newsgroups (20NG), Ohsumed, R52 and R8 of Reuters dataset, and Movie Review (MR). The original data can be found in https://github.com/yao8839836/text_gcn

## How to run
python train.py

## Dependencies
The script has been tested running under Python 3.6.2, with the following packages installed:

* numpy==1.14.1


