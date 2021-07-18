from inits import *
from utils import *

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

# topk function
def top_k(support_w,k=10,head_num=8):
    w=tf.transpose(support_w)
    lst=[]
    for i in range(head_num):
        tw = w[i, :]
        z = tf.zeros_like(tw)
        k_value,_ = tf.nn.top_k(tw,k,sorted=True)
        c = k_value[k-1]
        v= tf.where(tw>= c,tw, z)
        lst.append(v)
    r = tf.stack(lst)
    r = tf.transpose(r)
    return r

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

# MP-GCN-1
class GraphConvolution_MP_1(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, vocab_size,dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,is_cat=0,
                 featureless=False,head_num=1,name='0',p=0.01,**kwargs):
        super(GraphConvolution_MP_1, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.vocab_size=vocab_size
        # --------------------------
        self.p = p
        self.head_num=head_num
        self.is_cat=is_cat
        # --------------------------
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.name=name

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_o'+name] = glorot([input_dim, int(output_dim)], name='weights_o'+name)
            self.vars['d_weights_r'+name] = glorot([input_dim, self.head_num],name='d_weights_r'+name)
            self.vars['d_weights_s'+name] = glorot([self.head_num*output_dim,output_dim], name='d_weights_s'+name)
            if self.bias:
                self.vars['bias'+name] = zeros([output_dim], name='bias'+name)

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
            pre_sup = dot(x, self.vars['weights_o'+self.name],sparse=self.sparse_inputs)
            pre_sup_w = dot(x, self.vars['d_weights_r'+self.name],sparse=self.sparse_inputs)
        else:
            pre_sup = self.vars['weights_o'+self.name]
            pre_sup_w = self.vars['d_weights_r'+self.name]

        support = dot(self.support[0], pre_sup, sparse=True)

        #self-attention
        support_w = dot(self.support[0], pre_sup_w, sparse=True)
        support_w = tf.nn.tanh(support_w)
        support_w_ = tf.nn.softmax(support_w)

        #topk
        support_w=top_k(support_w_,int(self.vocab_size*self.p),self.head_num)
     
        #concat head
        enhances=[]
        for i in range(self.head_num):
            enhances.append(np.dot(support, tf.expand_dims(support_w[:,i], -1)))
        enhance=tf.concat(enhances,axis=1)
        e=tf.matmul(enhance,self.vars['d_weights_s'+self.name])
        #add
        output=(e+ support)/2
        #concat hidden state
        if self.is_cat==1:
            output=tf.concat([output,support],axis=1)
        # bias
        if self.bias:
            output += self.vars['bias'+self.name]
        self.embedding = output
        return self.act(output)

#MP-GCN-1*
class GraphConvolution_MP_Star(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, vocab_size,dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,is_cat=0,
                 featureless=False,head_num=8,name='0', p=0.01,**kwargs):
        super(GraphConvolution_MP_Star, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        # --------------------------
        self.p = p
        self.head_num=head_num
        self.is_cat=is_cat
        # --------------------------
        self.vocab_size = vocab_size
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.name = name

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_' + str(0)] = glorot([input_dim, output_dim], name='weights_' + str(0))#第一层1图
  
            #多头参数
            self.vars['d_weights_r'+ str(0)] = glorot([input_dim, self.head_num],name='d_weights_r'+ str(0))
            self.vars['d_weights_r' + str(1)] = glorot([input_dim, self.head_num], name='d_weights_r' + str(1))
            #self.vars['d_weights_r' + str(2)] = glorot([input_dim, self.head_num], name='d_weights_r' + str(2))
            #FC
            self.vars['d_weights_s'+ str(0)] = glorot([self.head_num*output_dim*2,output_dim], name='d_weights_s'+ str(0))
            if self.bias:
                self.vars['bias'] = zeros([2], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve------------------------------------------
        hiddens = list()
        # token embedding
        if not self.featureless:
            pre_sup = dot(x, self.vars['weights_' + str(0)],sparse=self.sparse_inputs)
        else:
            pre_sup = self.vars['weights_' + str(0)]
        support = dot(self.support[0], pre_sup, sparse=True)
        hiddens.append(support)

        # position embedding
        support_p = dot(self.support[1], pre_sup, sparse=True)
        hiddens.append(support_p)

        # semantic embedding
        #support_s = dot(self.support[1], pre_sup, sparse=True)
        #hiddens.append(support_s)

        # 初始化self-attention-----------------------------------------
        if not self.featureless:
            pre_sup_w = dot(x, self.vars['d_weights_r' + str(0)], sparse=self.sparse_inputs)
        else:
            pre_sup_w = self.vars['d_weights_r' + str(0)]

        # first graph self-attention
        support_w = dot(self.support[0], pre_sup_w, sparse=True)
        support_w = tf.nn.tanh(support_w)
        support_w = tf.nn.softmax(support_w)
        support_w=top_k(support_w,int(self.vocab_size*self.p),self.head_num)

        # second graph self-attention
        if not self.featureless:
           pre_sup_w = dot(x, self.vars['d_weights_r' + str(1)], sparse=self.sparse_inputs)
        else:
           pre_sup_w = self.vars['d_weights_r' + str(1)]
        position_w = dot(self.support[1], pre_sup_w, sparse=True)
        position_w = tf.nn.tanh(position_w)
        position_w = tf.nn.softmax(position_w)
        position_w=top_k(position_w,int(self.vocab_size*self.p),self.head_num)

        # third graph self-attention
        #if not self.featureless:
        #    pre_sup_w = dot(x, self.vars['d_weights_r' + str(2)], sparse=self.sparse_inputs)
        #else:
        #    pre_sup_w = self.vars['d_weights_r' + str(2)]
        #similiar_w = dot(self.support[2], pre_sup_w, sparse=True)
        #similiar_w = tf.nn.tanh(similiar_w)
        #similiar_w = tf.nn.softmax(similiar_w)
        #similiar_w =top_k(similiar_w,int(self.vocab_size*self.p),self.head_num)

        #concat-------------------------------------------------------
        enhances = []
        for i in range(self.head_num):
            enhances.append(np.dot(hiddens[0], tf.expand_dims(support_w[:, i], -1)))
        for i in range(self.head_num):
            enhances.append(np.dot(hiddens[1], tf.expand_dims(position_w[:, i], -1)))
        #for i in range(self.head_num):
        #    enhances.append(np.dot(hiddens[2], tf.expand_dims(similiar_w[:, i], -1)))

        enhance = tf.concat(enhances, axis=1)
        e = tf.matmul(enhance, self.vars['d_weights_s' + str(0)])
        #add
        output =support+e
        #concat hidden state
        if self.is_cat==1:
            output=tf.concat([output,support],axis=1)
        # bias
        if self.bias:
            output += self.vars['bias']
        self.embedding = output
        return self.act(output)

# MP-GCN-2
class GraphConvolution_MP_2(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, vocab_size, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,is_cat=0,
                 featureless=False, head_num=1, name='0', p=0.01, **kwargs):
        super(GraphConvolution_MP_2, self).__init__(**kwargs)

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
        # --------------------------
        self.p = p
        self.head_num=head_num
        self.is_cat=is_cat
        # --------------------------
        self.vocab_size = vocab_size
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_o' + name] = glorot([input_dim, int(output_dim)], name='weights_o' + name)
            self.vars['d_weights_r' + name] = glorot([input_dim, self.head_num], name='d_weights_r' + name)  # 多头注意力
            self.vars['d_weights_r2' + name] = glorot([output_dim, self.head_num], name='d_weights_r2' + name)  # 多头池化
            self.vars['d_weights_rf' + name] = glorot([self.head_num * output_dim * 2, output_dim], name='d_weights_rf' + name)

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
            pre_sup_w = dot(x, self.vars['d_weights_r' + self.name], sparse=self.sparse_inputs)
        else:
            pre_sup = self.vars['weights_o' + self.name]
            pre_sup_w = self.vars['d_weights_r' + self.name]

        support = dot(self.support[0], pre_sup, sparse=True)

        # 1-layer self-attention
        support_w = dot(self.support[0], pre_sup_w, sparse=True)
        support_w = tf.nn.tanh(support_w)
        support_w = tf.nn.softmax(support_w)
        support_w = top_k(support_w, int(self.vocab_size * self.p), self.head_num)

        # 2-layer self-attention
        support_w2 = dot(support, self.vars['d_weights_r2' + self.name], sparse=False)
        support_w2 = dot(self.support[0], support_w2, sparse=True)
        support_w2 = tf.nn.tanh(support_w2)
        support_w2 = tf.nn.softmax(support_w2)
        support_w2 = top_k(support_w2, int(self.vocab_size * self.p), self.head_num)
        # concat
        enhances = []
        for i in range(self.head_num):
            enhances.append(np.dot(support, tf.expand_dims(support_w[:, i], -1)))
        for i in range(self.head_num):
            enhances.append(np.dot(support, tf.expand_dims(support_w2[:, i], -1)))

        enhance = tf.concat(enhances, axis=1)
        e = tf.matmul(enhance, self.vars['d_weights_rf' + self.name])
        output = (e + support)/2

        #concat hidden state
        if self.is_cat==1:
            output=tf.concat([output,support],axis=1)

        # bias
        if self.bias:
            output += self.vars['bias' + self.name]
        self.embedding = output
        return self.act(output)
