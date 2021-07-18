from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from utils import *
from models import GCN, MP_GCN
import os
from metrics import classify_evalue


dataset = 'mr'#'20ng', 'R8', 'R52', 'ohsumed', 'mr'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
# Set the maximum GPU occupied not more than 80% of the video memory
config.gpu_options.per_process_gpu_memory_fraction = 0.8
# Set up dynamic GPU allocation
config.gpu_options.allow_growth = True

#set seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', dataset, 'Dataset string.')
# 'gcn', 'mp_gcn'
flags.DEFINE_string('model', 'mp_gcn', 'Model string.')
#------------------------------------
flags.DEFINE_integer('type', 0, 'Model type') #0:mp-gcn-1 | 1:mp-gcn-1* | 2:mp-gcn-2
flags.DEFINE_float('ratio', 0.2, 'Topk params') # mr 0.2 | others 0.01
flags.DEFINE_integer('head_num', 8, 'Head number')
flags.DEFINE_integer('is_cat', 0, 'Concatenation hidden state')  #0 no /1 yes
#------------------------------------
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('epochs',1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200 , 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')  # 0
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).') #mr:10-30 | others: 30-50
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

#Read different graph data
path_name='data'
if FLAGS.type==1:
    path_name = 'data_m'

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size,vocab_size = load_corpus(FLAGS.dataset,file_name=path_name)
nb_nodes=features.shape[0]# V
features = sp.identity(nb_nodes)  # featureless

# Some preprocessing
features = preprocess_features(features)

# Initialize adjacency matrix
if isinstance(adj, list):
    support = [preprocess_adj(adj[0]), preprocess_adj(adj[1])] #multi-graph
    num_supports = len(support)
else:
    support = [preprocess_adj(adj)] #single-graph
    num_supports = 1

# Set model
if FLAGS.model == 'gcn':
    model_func = GCN
elif FLAGS.model == 'mp_gcn':
    model_func = MP_GCN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': tf.placeholder(tf.int32),
}

# Input params
input_dict={
    'input_dim':features[2][1],
    'train_size': train_size,
    'vocab_size': vocab_size,
    'head_num':FLAGS.head_num,
    'type': FLAGS.type,
    'p': FLAGS.ratio,
    'is_cat':FLAGS.is_cat,
}
# Create model
model = model_func(placeholders, input_dim=input_dict, logging=True)
# Initialize session
sess = tf.Session(config=config)
# Init variables
sess.run(tf.global_variables_initializer())
cost_val = []

# -----------------------------functions-------------------------------
# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(
        features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test)

# -----------------------------train-------------------------------
# Train model
for epoch in range(FLAGS.epochs):
    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(
        features, support,y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.layers[0].embedding], feed_dict=feed_dict)
    # Validation
    cost, acc, pred, labels, duration = evaluate(
        features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(
              outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# -----------------------------test-------------------------------
# Testing
test_cost, test_acc, pred, labels, test_duration = evaluate(
    features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

test_labels = []
for i in range(len(test_mask)):
    if test_mask[i]:
        test_labels.append(labels[i])

test_pred = pred[train_size+vocab_size:].tolist()

if len(test_pred)==len(test_labels):
    acc_r=classify_evalue(test_labels, test_pred, 'acc')
    f1_r=classify_evalue(test_labels, test_pred, 'f1')
    roc_auc_r=classify_evalue(test_labels, test_pred, 'roc_auc')
    print(acc_r)
    print(f1_r)
    print(roc_auc_r)

# Get graph embeddings
word_embeddings = outs[3][train_size: nb_nodes - test_size]
train_doc_embeddings = outs[3][:train_size]  # include val docs
test_doc_embeddings = outs[3][nb_nodes - test_size:]

# Save graph embeddings
# write_vector(dataset,word_embeddings,train_doc_embeddings,test_doc_embeddings,train_size,test_size)
