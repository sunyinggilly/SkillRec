import tensorflow as tf
import numpy as np
from math import log


class ActDNN(object):
    def __init__(self, n_s, verbose=1, lr=0.1, deep_layers=[16, 16], dropout_deep=[1.0, 1.0, 1.0],
                 activation=tf.nn.relu, random_seed=2019, l2_reg=0.0):
        self.n_s = n_s
        self.random_seed = random_seed
        self.lr = lr
        self.verbose = verbose
        self.activation = activation
        self.l2_reg = l2_reg
        self.deep_layers = deep_layers
        self.dropout_deep_feed = dropout_deep
        self.weights = {}
        self.dropouts = {}
        self._init_graph()

    def get_dict(self, data, train=True):
        feed_dict = {
            self.input_state: data[0],
            self.input_action: data[1],
            self.deep_dropout: self.dropout_deep_feed if train else [1] * len(self.dropout_deep_feed)
        }
        return feed_dict

    def run(self, data, batch_size, train=True):
        n_data = len(data[0])
        predictions = []
        for i in range(0, n_data, batch_size):
            data_batch = [dt[i:min(i + batch_size, n_data)] for dt in data]
            if train:
                preds, loss, _ = self.sess.run((self.pred, self.loss, self.optimizer), feed_dict=self.get_dict(data_batch, train=True))
            else:
                preds = self.sess.run(self.pred, feed_dict=self.get_dict(data_batch, train=False))
            predictions.extend(preds)
        return predictions

    def _count_parameters(self, print_count=False):
        total_parameters = 0
        for name, variable in self.weights.items():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            if print_count:
                print(name, variable_parameters)
            total_parameters += variable_parameters
        return total_parameters

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self._build_network()
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=1e-8)\
                               .minimize(self.loss)
            # init
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.saver = tf.train.Saver(max_to_keep=1)
            init = tf.global_variables_initializer()
            self.sess = tf.Session(config=config)
            self.sess.run(init)
            print("#params: %d" % self._count_parameters())

    def _build_network(self):
        # 输入
        self.input_state = tf.placeholder(tf.float32, shape=[None, self.n_s], name="input_state")
        self.input_action = tf.placeholder(tf.int32, shape=[None], name="input_action")
        self.onehot_action = tf.one_hot(self.input_action, self.n_s)
        self.q, penalty_loss = self._state_action_model(self.input_state, self.onehot_action)

        self.q = self.q - self.input_state * 1000
        self.q = tf.nn.softmax(self.q, axis=1)
        self.q = tf.multiply(self.q, tf.ones_like(self.input_state) - self.input_state)
        self.q = self.q / tf.reshape(tf.reduce_sum(self.q, axis=1), [-1, 1])
        self.v = tf.reduce_max(self.q, axis=1)

        self.loss = tf.losses.log_loss(predictions=self.q, labels=self.onehot_action)
        self.loss += penalty_loss
        self.pred = tf.reduce_sum(tf.multiply(self.q, self.onehot_action), axis=1)
        return

    def _state_action_model(self, input_state, onehot_action):
        deep_in = input_state
        self.deep_dropout, y_deep, loss = self._mlp(deep_in, self.n_s, self.deep_layers, name="state_deep",
                                                    activation=self.activation, bias=True, sparse_input=False)
        q, _ = self._fc(y_deep, self.deep_layers[-1], self.n_s, name="output", l2_reg=0.0,
                        activation=None, bias=True, sparse=False)
        return q, loss

    def _fc(self, tensor, dim_in, dim_out, name, l2_reg, activation=None, bias=True, sparse=False):
        glorot = np.sqrt(2.0 / (dim_in + dim_out))
        if "%s_w" % name not in self.weights:
            self.weights["%s_w" % name] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(dim_in, dim_out)),
                                                      dtype=np.float32, name="%s_w" % name)
        if not sparse:
            y_deep = tf.matmul(tensor, self.weights["%s_w" % name])
        else:
            y_deep = tf.sparse_tensor_dense_matmul(tensor, self.weights["%s_w" % name])
        if bias:
            if "%s_b" % name not in self.weights:
                self.weights["%s_b" % name] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, dim_out)),
                                                          dtype=np.float32, name="%s_b" % name)
                y_deep += self.weights["%s_b" % name]
        if activation is not None:
            y_deep = activation(y_deep)
        return y_deep, tf.contrib.layers.l2_regularizer(l2_reg)(self.weights["%s_w" % name])

    def _mlp(self, tensor, dim_in, layers, name, activation=None, bias=True, sparse_input=False):
        if name not in self.dropouts:
            self.dropouts[name] = tf.placeholder(tf.float32, shape=[None], name="%s_dropout" % name)
        dropout = self.dropouts[name]
        y_deep = tf.nn.dropout(tensor, dropout[0])
        lst = []
        loss = 0
        for i, layer in enumerate(layers):
            if i == 0 and sparse_input:
                y_deep, loss_now = self._fc(y_deep, dim_in, layer, l2_reg=self.l2_reg, name="%s_%d" % (name, i), bias=bias,
                                            activation=activation, sparse=True)
            else:
                y_deep, loss_now = self._fc(y_deep, dim_in, layer, l2_reg=self.l2_reg, name="%s_%d" % (name, i), bias=bias,
                                            activation=activation, sparse=False)
            y_deep = tf.nn.dropout(y_deep, dropout[i + 1])
            loss += loss_now
            lst.append(y_deep)
            dim_in = layer
        return dropout, y_deep, loss

    def save(self, save_path):
        self.saver.save(self.sess, save_path)

    def load(self, load_path):
        self.saver.restore(self.sess, load_path)

    def evaluate_metrics(self, y_pred):
        ll = -sum([log(u) for u in y_pred]) / len(y_pred)
        return [("ll", ll)]

    def predict(self, data, batch_size=32):
        predictions, _ = self.run(data, batch_size, train=False)
        return predictions

    def print_result(self, data_eval, endch="\n"):
        print_str = ""
        for i, name_val in enumerate(data_eval):
            if i != 0:
                print_str += ','
            print_str += "%s: %f" % name_val
        print(print_str, end=endch)

    def estimate_maxq_action(self, state_vis, act_pool=None):
        q_ret, s_ret = -1000000000, 0
        data = [[state_vis], [0], [0]]
        q = self.sess.run(self.q, self.get_dict(data, train=False))[0]
        if act_pool is None:
            act_pool = list(range(self.n_s))
        for i in act_pool:
            if state_vis[i] == 1:
                continue
            q_now = q[i]
            if q_ret < q_now:
                q_ret, s_ret = q_now, i
        return q_ret, s_ret

    def estimate_single(self, skill_vis, action):
        data = [[skill_vis], [action], [0]]
        return self.predict(data)[0]

    def get_policy(self, state_vis):
        data = [[state_vis], [0], [0]]
        p = self.sess.run(self.q, self.get_dict(data, train=False))[0]
        return p
