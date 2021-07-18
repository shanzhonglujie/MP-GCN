import os
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from utils import loadWord2Vec
from math import log
import sys
# import networkx as nx
# from sklearn import svm
# from nltk.corpus import wordnet as wn
# from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy.spatial.distance import cosine

def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

# Semantic distance
def get_similiar(word_pair_count):
    word_vector_file = path_name+'/glove.6B/glove.6B.300d.txt'
    vocab, data, word_vector_map = loadWord2Vec(word_vector_file)
    word_similiar={}
    for word_pair_str in word_pair_count:
        pair=word_pair_str.split(',')
        vector_i = np.array(word_vector_map[vocab[int(pair[0])]])
        vector_j = np.array(word_vector_map[vocab[int(pair[1])]])
        similarity =cos_sim(vector_i,vector_j)
        word_similiar[word_pair_str]=similarity
    return word_similiar

# if len(sys.argv) != 2:
# 	sys.exit("Use: python build_graph.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
# build corpus
dataset = 'mr'
path_name = 'data_m'

if dataset not in datasets:
	sys.exit("wrong dataset name")

# Read Word Vectors
#word_vector_file = path_name+'/glove.6B/glove.42B.300d.txt'
#word_vector_file = path_name+'/corpus/' + dataset + '_word_vectors.txt'
#word_vector_file = path_name+'/' + dataset + '_word_vectors.txt'
#_, embd, word_vector_map = loadWord2Vec(word_vector_file)
#word_embeddings_dim = len(embd[0])

word_embeddings_dim = 300
word_vector_map = {}

# shulffing
doc_name_list = []
doc_train_list = []
doc_test_list = []

f = open(path_name+'/' + dataset + '.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_name_list.append(line.strip())
    temp = line.split("\t")
    if temp[1].find('test') != -1:
        doc_test_list.append(line.strip())
    elif temp[1].find('train') != -1:
        doc_train_list.append(line.strip())
f.close()

doc_content_list = []
f = open(path_name+'/corpus/' + dataset + '.clean.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_content_list.append(line.strip())
f.close()
# print(doc_content_list)

train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
# print(train_ids)
random.shuffle(train_ids)

# partial labeled data
train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open(path_name+'/' + dataset + '.train.index', 'w')
f.write(train_ids_str)
f.close()

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
# print(test_ids)
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open(path_name+'/' + dataset + '.test.index', 'w')
f.write(test_ids_str)
f.close()

ids = train_ids + test_ids
# print(ids)
# print(len(ids))

shuffle_doc_name_list = []
shuffle_doc_words_list = []
for id in ids:
    shuffle_doc_name_list.append(doc_name_list[int(id)])
    shuffle_doc_words_list.append(doc_content_list[int(id)])
shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

f = open(path_name+'/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_name_str)
f.close()

f = open(path_name+'/corpus/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_words_str)
f.close()

# build vocab
word_freq = {}
word_set = set()
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)

word_doc_list = {}
for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)
f = open(path_name+'/corpus/' + dataset + '_vocab.txt', 'w')
f.write(vocab_str)
f.close()

# label list
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)

label_list_str = '\n'.join(label_list)
f = open(path_name+'/corpus/' + dataset + '_labels.txt', 'w')
f.write(label_list_str)
f.close()

# x: feature vectors of training docs, no initial features
# slect 90% training set
train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size  # - int(0.5 * train_size)

# different training rates
real_train_doc_names = shuffle_doc_name_list[:real_train_size]
real_train_doc_names_str = '\n'.join(real_train_doc_names)

f = open(path_name+'/' + dataset + '.real_train.name', 'w')
f.write(real_train_doc_names_str)
f.close()

row_x = []
col_x = []
data_x = []
for i in range(real_train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            # print(doc_vec)
            # print(np.array(word_vector))
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

# x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    real_train_size, word_embeddings_dim))

y = []
for i in range(real_train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)
# print(y)

# tx: feature vectors of test docs, no initial features
test_size = len(test_ids)

row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i + train_size]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

# tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))

ty = []
for i in range(test_size):
    doc_meta = shuffle_doc_name_list[i + train_size]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)
# print(ty)

# allx: the the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> words

word_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, word_embeddings_dim))

for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []

for i in range(train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))


row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

ally = []
for i in range(train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)

print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

'''
Doc word heterogeneous graph
'''

# word co-occurence with context windows
window_size = 20
windows = []

for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)

word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])


#获得词对出现的次数与距离
word_pair_count = {}
word_pair_distant = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
                if i!=j:
                    word_pair_distant[word_pair_str]+=i-j
            else:
                word_pair_count[word_pair_str] = 1
                if i != j:
                    word_pair_distant[word_pair_str] =i-j
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
                if i != j:
                    word_pair_distant[word_pair_str] += j-i
            else:
                word_pair_count[word_pair_str] = 1
                if i != j:
                    word_pair_distant[word_pair_str] = j-i

row = []
col = []
weight = []

# pmi as weights
num_window = len(windows)
for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        word_pair_distant.pop(key)
        continue

    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)
print('calculate pmi')

#calculate distance------------------------------------------------
row2 = []
col2 = []
weight2 = []
# distant as weights
for key in word_pair_distant:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    if word_pair_distant[key]==0:
        mean_distant=word_pair_count[key]
    else:
        mean_distant=word_pair_count[key]/(word_pair_distant[key])
    row2.append(train_size + i)
    col2.append(train_size + j)
    weight2.append(mean_distant)
print('calculate distanct')

#获得语义
# word_similiar=get_similiar(word_pair_distant)
# row3 = []
# col3 = []
# weight3 = []
# # distant as weights
# for key in word_similiar:
#     temp = str(key).split(',')
#     i = int(temp[0])
#     j = int(temp[1])
#     row3.append(train_size + i)
#     col3.append(train_size + j)
#     weight3.append(word_similiar[key])
# print('create similiar nodes')


# doc word frequency
doc_word_freq = {}

for doc_id in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row.append(i)
            row2.append(i)
            # row3.append(i)
        else:
            row.append(i + vocab_size)
            row2.append(i + vocab_size)
            # row3.append(i + vocab_size)
        col.append(train_size + j)
        col2.append(train_size + j)
        # col3.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) /
                  word_doc_freq[vocab[j]])
        weight.append(freq * idf)
        weight2.append(1)
        # weight3.append(1)
        doc_word_set.add(word)

node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))
adj2 = sp.csr_matrix((weight2, (row2, col2)), shape=(node_size, node_size))
# adj3 = sp.csr_matrix((weight3, (row3, col3)), shape=(node_size, node_size))

adjs=[]
adjs.append(adj)
adjs.append(adj2)
# adjs.append(adj3)
print('create nodes (window_size='+str(window_size)+')')

# dump objects
f = open(path_name+"/ind.{}.x".format(dataset), 'wb')
pkl.dump(x, f)
f.close()

f = open(path_name+"/ind.{}.y".format(dataset), 'wb')
pkl.dump(y, f)
f.close()

f = open(path_name+"/ind.{}.tx".format(dataset), 'wb')
pkl.dump(tx, f)
f.close()

f = open(path_name+"/ind.{}.ty".format(dataset), 'wb')
pkl.dump(ty, f)
f.close()

f = open(path_name+"/ind.{}.allx".format(dataset), 'wb')
pkl.dump(allx, f)
f.close()

f = open(path_name+"/ind.{}.ally".format(dataset), 'wb')
pkl.dump(ally, f)
f.close()

f = open(path_name+"/ind.{}.adj".format(dataset), 'wb')
pkl.dump(adjs, f)
f.close()
