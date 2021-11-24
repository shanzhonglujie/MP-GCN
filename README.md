# MP-GCN
H. Zhao, J. Xie, H. Wang, "Graph Convolutional Network based on Multi-head Pooling for Short Text Classification".

## Overview
Here we provide the implementation of a MP-GCN in TensorFlow, along with a minimal execution example (on the MR dataset). The repository is organised as follows:

* data/  # contains the necessary dataset files for MR
* test/  # contains some baselines
* vector/  # is used to store graph embeddings
* build_multi_graph.py  # is used to create multi-graph to verify MP-GCN-1*
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
* Create empty folders 'data_m' and 'vector'
* python build_single_graph.py  # create graph
* python train.py  # training and testing

## Dependencies
The script has been tested running under Python 3.6.5 and tensorflow 1.8.0
