# -*- coding: utf-8 -*-
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import numpy as np


class DQN(object):
    def __init__(self, n_s, lambda_d=0.01, verbose=1, lr=0.1, embsize=16, pool_size=50,
                 easy_deep_layers=[16, 16], dropout_easy_deep=[1.0, 1.0, 1.0],
                 salary_deep_layers=[16, 16], dropout_salary_deep=[1.0, 1.0, 1.0],
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
        self.easy_deep_layers = easy_deep_layers
        self.salary_deep_layers = salary_deep_layers
        self.dropout_easy_deep_feed = dropout_easy_deep
        self.dropout_salary_deep_feed = dropout_salary_deep

        self.lambda_d = lambda_d
        self.weights = {}
        self.dropouts = {}
        self._init_graph()

    def get_dict(self, data, train=True):
        feed_dict = {
            self.input_state: data[0],
            self.input_actpool: data[1],
            self.salary_label: data[2],
            self.easy_label: data[3],
            self.deep_easy_dropout: self.dropout_easy_deep_feed if train else [1] * len(self.dropout_easy_deep_feed),
            self.deep_salary_dropout: self.dropout_salary_deep_feed if train else [1] * len(self.dropout_salary_deep_feed),
        }
        return feed_dict

    def run(self, data, batch_size, train=True):
        n_data = len(data[0])
        predictions_salary, predictions_easy = [], []
        for i in range(0, n_data, batch_size):
            data_batch = [dt[i:min(i + batch_size, n_data)] for dt in data]
            if train:
                preds_easy, preds_salary, loss, _ = self.sess.run(
                    (self.v_easy, self.v_salary, self.loss, self.optimizer),
                    feed_dict=self.get_dict(data_batch, train=True))

            else:
                preds_easy, preds_salary = self.sess.run((self.v_easy, self.v_salary),
                                                         feed_dict=self.get_dict(data_batch, train=False))
            predictions_salary.extend(preds_salary)
            predictions_easy.extend(preds_easy)

        return predictions_salary, predictions_easy

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
        self.salary_label = tf.placeholder(tf.float32, shape=[None], name="%s_input_salary_label" % self.name)
        self.easy_label = tf.placeholder(tf.float32, shape=[None], name="%s_input_easy_label" % self.name)
        self.input_actpool = tf.placeholder(tf.int32, shape=[None, self.pool_size], name="%s_input_actpool" % self.name)

        # ------------------------ 读embedding -------------------
        emb = tf.Variable(np.random.normal(size=(self.n_s, self.embsize)),
                          dtype=np.float32, name="%s_embedding" % self.name)
        self.weights["%s_embedding" % self.name] = emb

        input_emb = tf.matmul(self.input_state, emb)
        input_emb = tf.tile(tf.reshape(input_emb, [-1, 1, self.embsize]), [1, self.pool_size, 1])
        act_emb = tf.nn.embedding_lookup(emb, self.input_actpool)

        # ------------------------ salary部分 ---------------------
        salary_in = tf.concat((input_emb, act_emb), axis=2)  # -1, pool_size, embsize
        deep_in = tf.reshape(salary_in, [-1, self.embsize * 2])

        self.deep_salary_dropout, y_deep, penalty_loss = self._mlp(deep_in, self.embsize * 2, self.salary_deep_layers,
                                                                   name="%s_deep_salary" % self.name,
                                                                   activation=self.activation, bias=True, sparse_input=False)
        self.q_salary, _ = self._fc(y_deep, self.salary_deep_layers[-1], 1, name="%s_salary_Inc" % self.name, l2_reg=0.0,
                                    activation=None, bias=True, sparse=False)

        self.q_salary = tf.reshape(self.q_salary, [-1, self.pool_size])

        # --------------------- easy部分 --------------------------------
        easy_in = tf.concat((input_emb, act_emb), axis=2)  # -1, pool_size, embsize
        deep_in = tf.reshape(easy_in, [-1, self.embsize * 2])

        self.deep_easy_dropout, y_deep, loss = self._mlp(deep_in, self.embsize * 2, self.easy_deep_layers,
                                                         name="%s_deep_easy" % self.name,
                                                         activation=self.activation, bias=True, sparse_input=False)
        penalty_loss += loss
        self.q_easy, _ = self._fc(y_deep, self.easy_deep_layers[-1], 1, name="%s_easy_output" % self.name, l2_reg=0.0,
                                  activation=None, bias=True, sparse=False)
        self.q_easy = tf.reshape(self.q_easy, [-1, self.pool_size])

        # --------------------- action预测 ------------------------------
        self.q = self.lambda_d * self.q_easy + self.q_salary
        self.v_place = tf.argmax(self.q, 1)

        onehot_action_nxt = tf.one_hot(self.v_place, self.pool_size)  # -1, M
        self.v_skill = tf.cast(tf.reduce_sum(tf.multiply(onehot_action_nxt, tf.cast(self.input_actpool, np.float32)), 1), np.int32)
        self.v = tf.reduce_sum(tf.multiply(onehot_action_nxt, self.q), 1)
        self.v_salary = tf.reduce_sum(tf.multiply(onehot_action_nxt, self.q_salary), 1)
        self.v_easy = tf.reduce_sum(tf.multiply(onehot_action_nxt, self.q_easy), 1)

        # --------------------- loss ----------------------------------
        self.loss = tf.reduce_mean(tf.square(self.v_salary - self.salary_label))
        self.loss += tf.reduce_mean(tf.square(self.v_easy - self.easy_label))
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

    def evaluate_metrics(self, y_pred_salary, y_true_salary, y_pred_easy, y_true_easy):
        salary_mse = mean_squared_error(y_true_salary, y_pred_salary)
        easy_mse = mean_squared_error(y_true_easy, y_pred_easy)
        return [("salary_mse", salary_mse), ("easy_mse", easy_mse)]

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
        act, v_salary, v_easy = self.sess.run((self.v_skill, self.v_salary, self.v_easy), self.get_dict(data, train=False))
        return (v_salary[0], v_easy[0]), act[0]

    def estimate_maxq_batch(self, data_state, data_pool):
        n_data = len(data_state)
        act_lst, v_salary, v_easy = self.sess.run((self.v_skill, self.v_salary, self.v_easy),
                                                  self.get_dict((data_state, data_pool, [0] * n_data, [0] * n_data), train=False))
        return (v_salary, v_easy), act_lst

    def get_q_list(self, skill_vis, act_pool):
        data = [[skill_vis], [act_pool], [0], [0]]
        q_salary_lst, q_easy_lst = self.sess.run((self.q_salary, self.q_easy), self.get_dict(data, train=False))
        return q_salary_lst[0], q_easy_lst[0]
