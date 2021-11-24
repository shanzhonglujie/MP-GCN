from inits import *
from utils import *
import random

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

# topk function
def ntop_k(support_w,k,head_num):
    # RS operation in this paper has the best effect when collecting balanced samples
    # take the front head_num dimensions, exactly corresponding to the balanced sample
    w=tf.transpose(support_w)
    lst=[]
    for i in range(head_num):
        tw = w[i, :]
        z = tf.zeros_like(tw)
        o = tf.ones_like(tw)
        k_value,_ = tf.nn.top_k(tw,k,sorted=True)
        c = k_value[k-1]
        v= tf.where(tw>= c,o, z)
        lst.append(v)
    r = tf.stack(lst)
    z_max = tf.reduce_max(r,axis=0)
    z_mean = tf.reduce_mean(r,axis=0)
    r_max = tf.transpose(z_max)
    r_mean = tf.transpose(z_mean)
    return r_max,r_mean

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

# Define the base class
class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

# GCN
class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)
        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]

            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)
        # bias
        if self.bias:
            output += self.vars['bias']
        self.embedding = output
        return self.act(output)

# MP-GCN
class GraphConvolution_MP(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, vocab_size, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, head_num=1, name='0', p=0.01, **kwargs):
        super(GraphConvolution_MP, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.name = name
        self.p = p
        self.head_num=head_num
        self.vocab_size = vocab_size
        self.output_dim=output_dim
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_o' + name] = glorot([input_dim, output_dim], name='weights_o' + name)
            if self.bias:
                self.vars['bias' + name] = zeros([output_dim], name='bias' + name)

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)
        # convolve
        if not self.featureless:
            pre_sup = dot(x, self.vars['weights_o' + self.name], sparse=self.sparse_inputs)
        else:
            pre_sup = self.vars['weights_o' + self.name]
        
        pre_sup = tf.nn.dropout(pre_sup, 1 - self.dropout)
        #gcnconv
        support = dot(self.support[0], pre_sup, sparse=True)
        # tf-idf
        support_w = dot(self.support[1], pre_sup, sparse=True)
        #support_w = tf.nn.tanh(support_w)
        #support_w = tf.nn.softmax(support_w)
        support_w = tf.nn.sigmoid(support_w)
        wmax,wmean = ntop_k(support_w, int(self.vocab_size * self.p), self.head_num)
        # pmi        
        support_w2 = dot(self.support[2], pre_sup, sparse=True)
        #support_w2 = tf.nn.tanh(support_w2)
        #support_w2 = tf.nn.softmax(support_w2)
        support_w2 = tf.nn.sigmoid(support_w2)
        wmax2,wmean2 = ntop_k(support_w2, int(self.vocab_size * self.p), self.head_num)
        # add
        wmax=tf.add(wmax,wmax2)
        wmean=tf.add(wmean,wmean2)
        # add & weight
        output = (support+np.dot(support, tf.expand_dims(wmax,-1)) + np.dot(support, tf.expand_dims(wmean,-1)))

        # bias
        if self.bias:
            output += self.vars['bias' + self.name]
        self.embedding = output
        return self.act(output)
