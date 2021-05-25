# -*- coding: utf-8 -*-
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import numpy as np


class DQNSi(object):
    def __init__(self, n_s, lambda_d=0.01, verbose=1, lr=0.1, embsize=16, pool_size=50,
                 deep_layers=[16, 16], dropout_deep=[1.0, 1.0, 1.0],
                 activation=tf.nn.relu, random_seed=2019, l2_reg=0.0, name="", sess=None):

        self.n_s = n_s
        self.random_seed = random_seed
        self.lr = lr
        self.verbose = verbose
        self.activation = activation
        self.l2_reg = l2_reg
        self.sess = sess
        self.name = name
        self.pool_size = pool_size

        self.embsize = embsize
        self.deep_layers = deep_layers
        self.dropout_deep_feed = dropout_deep

        self.lambda_d = lambda_d
        self.weights = {}
        self.dropouts = {}
        self._init_graph()

    def get_dict(self, data, train=True):
        feed_dict = {
            self.input_state: data[0],
            self.input_actpool: data[1],
            self.label: data[2],
            self.deep_dropout: self.dropout_deep_feed if train else [1] * len(self.dropout_deep_feed),
            self.deep_base_dropout: self.dropout_deep_feed if train else [1] * len(self.dropout_deep_feed)
        }
        return feed_dict

    def run(self, data, batch_size, train=True):
        n_data = len(data[0])
        predictions = []
        for i in range(0, n_data, batch_size):
            data_batch = [dt[i:min(i + batch_size, n_data)] for dt in data]
            if train:
                preds, loss, _ = self.sess.run(
                    (self.v, self.loss, self.optimizer),
                    feed_dict=self.get_dict(data_batch, train=True))

            else:
                preds = self.sess.run(self.v, feed_dict=self.get_dict(data_batch, train=False))
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
        tf.set_random_seed(self.random_seed)
        self._build_network()
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss)
        # init
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.saver = tf.train.Saver(max_to_keep=1)
        init = tf.global_variables_initializer()
        if self.sess is None:
            self.sess = tf.Session(config=config)
            self.sess.run(init)
        print("#params: %d" % self._count_parameters())

    def _build_network(self):
        # 输入
        self.input_state = tf.placeholder(tf.float32, shape=[None, self.n_s], name="%s_input_state" % self.name)
        self.label = tf.placeholder(tf.float32, shape=[None], name="%s_input_label" % self.name)
        self.input_actpool = tf.placeholder(tf.int32, shape=[None, self.pool_size], name="%s_input_actpool" % self.name)

        # ------------------------ embedding -------------------
        emb = tf.Variable(np.random.normal(size=(self.n_s, self.embsize)),
                          dtype=np.float32, name="%s_embedding" % self.name)
        self.weights["%s_embedding" % self.name] = emb

        input_emb = tf.matmul(self.input_state, emb)
        input_emb = tf.tile(tf.reshape(input_emb, [-1, 1, self.embsize]), [1, self.pool_size, 1])
        act_emb = tf.nn.embedding_lookup(emb, self.input_actpool)

        # ------------------------ salary /easy part ---------------------
        salary_in = tf.concat((input_emb, act_emb), axis=2)  # -1, pool_size, embsize
        deep_in = tf.reshape(salary_in, [-1, self.embsize * 2])

        self.deep_dropout, y_deep, penalty_loss = self._mlp(deep_in, self.embsize * 2, self.deep_layers,
                                                            name="%s_deep_salary" % self.name,
                                                            activation=self.activation, bias=True, sparse_input=False)
        self.s_salary, _ = self._fc(y_deep, self.deep_layers[-1], 1, name="%s_salary_Inc" % self.name, l2_reg=0.0,
                                    activation=None, bias=True, sparse=False)

        salary_state_nxt = input_emb + act_emb
        deep_in = tf.reshape(salary_state_nxt, [-1, self.embsize])
        self.deep_base_dropout, y_deep, penalty_loss = self._mlp(deep_in, self.embsize, self.deep_layers,
                                                                 name="%s_deep_salary_base" % self.name,
                                                                 activation=self.activation, bias=True,
                                                                 sparse_input=False)

        self.b_salary_nxt, _ = self._fc(y_deep, self.deep_layers[-1], 1, name="%s_salary_qbase" % self.name, l2_reg=0.0,
                                        activation=None, bias=True, sparse=False)
        self.q = self.b_salary_nxt + self.s_salary
        self.q = tf.reshape(self.q, [-1, self.pool_size])

        # --------------------- action ------------------------------
        self.v_place = tf.argmax(self.q, 1)

        onehot_action_nxt = tf.one_hot(self.v_place, self.pool_size)  # -1, M
        self.v_skill = tf.cast(tf.reduce_sum(tf.multiply(onehot_action_nxt, tf.cast(self.input_actpool, np.float32)), 1), np.int32)
        self.v = tf.reduce_sum(tf.multiply(onehot_action_nxt, self.q), 1)

        # --------------------- loss ----------------------------------
        self.loss = tf.reduce_mean(tf.square(self.v - self.label))
        self.loss += penalty_loss
        return

    def _fc(self, tensor, dim_in, dim_out, name, l2_reg, activation=None, bias=True, sparse=False):
        glorot = np.sqrt(2.0 / (dim_in + dim_out))
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
                y_deep, loss_now = self._fc(y_deep, dim_in, layer, l2_reg=self.l2_reg, name="%s_%d" % (name, i),
                                            bias=bias, activation=activation, sparse=True)
            else:
                y_deep, loss_now = self._fc(y_deep, dim_in, layer, l2_reg=self.l2_reg, name="%s_%d" % (name, i),
                                            bias=bias, activation=activation, sparse=False)
            y_deep = tf.nn.dropout(y_deep, dropout[i + 1])
            lst.append(y_deep)
            dim_in = layer
        return dropout, y_deep, loss

    def save(self, save_path):
        self.saver.save(self.sess, save_path)

    def load(self, load_path):
        self.saver.restore(self.sess, load_path)

    def evaluate_metrics(self, y_pred, y_true):
        mse = mean_squared_error(y_true, y_pred)
        return [("mse", mse)]

    def predict(self, data, batch_size=32):
        predictions_salary, predictions_easy = self.run(data, batch_size, train=False)
        return predictions_salary, predictions_easy

    def print_result(self, data_eval, endch="\n"):
        print_str = ""
        for i, name_val in enumerate(data_eval):
            if i != 0:
                print_str += ','
            print_str += "%s: %f" % name_val
        print(print_str, end=endch)

    def estimate_maxq_action(self, state_vis, act_pool):
        data = [[state_vis], [act_pool], [0], [0]]
        act, v = self.sess.run((self.v_skill, self.v), self.get_dict(data, train=False))
        return v[0], act[0]

    def estimate_maxq_batch(self, data_state, data_pool):
        n_data = len(data_state)
        act_lst, v = self.sess.run((self.v_skill, self.v),
                                   self.get_dict((data_state, data_pool, [0] * n_data, [0] * n_data), train=False))
        return v, act_lst

    def get_q_list(self, skill_vis, act_pool):
        data = [[skill_vis], [act_pool], [0], [0]]
        q_lst = self.sess.run(self.q, self.get_dict(data, train=False))
        return q_lst[0]
