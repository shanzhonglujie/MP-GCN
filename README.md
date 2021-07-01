# MP-GCN
H. Zhao, J. Xie, H. Wang, "Graph Convolution Network based on Multi-head Pooling for Short Text Classifification",2021.

## Overview
Here we provide the implementation of a MP-GCN in TensorFlow, along with a minimal execution example (on the MR dataset). The repository is organised as follows:

data/ contains the necessary dataset files for Cora;
models/ contains the implementation of the GAT network (gat.py);
pre_trained/ contains a pre-trained Cora model (achieving 84.4% accuracy on the test set);
utils/ contains:
an implementation of an attention head, along with an experimental sparse version (layers.py);
preprocessing subroutines (process.py);
preprocessing utilities for the PPI benchmark (process_ppi.py).

Finally, execute_cora.py puts all of the above together and may be used to execute a full training run on Cora.

## Datesets
We ran our experiments on five widely used benchmark corpora including 20-Newsgroups (20NG), Ohsumed, R52 and R8 of Reuters dataset, and Movie Review (MR). The original data can be found in https://github.com/yao8839836/text_gcn

## Dependencies
The script has been tested running under Python 3.6.2, with the following packages installed:

* numpy==1.14.1


