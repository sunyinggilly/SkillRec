import os
import sys
sys.path.append("../../")
sys.path.append("../")
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from math import log

from CONFIG import HOME_PATH

from Environment.Environment import Environment
from Environment.JobMatcherLinux import JobMatcher
from Environment.DifficultyEstimatorGLinux import DifficultyEstimator

from Utils.JobReader import n_skill, sample_info, read_offline_samples, read_skill_graph, itemset_process
from Utils.Functions import evaluate
from Utils.Utils import sparse_to_dense, read_pkl_data, print_result
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from Sampler import BestStrategyPoolSampler
from Models.ActDNN import ActDNN


class ActTrainer(object):
    def __init__(self, Qa, environment, train_samples, pool_size):
        self.environment = environment
        self.sampler = BestStrategyPoolSampler(relation_lst, Qa, n_skill, pool_size)
        self.Qa = Qa
        self.train_samples = train_samples

    def train(self, n_batch, verbose_batch, batch_size, data_train):
        data_train, data_valid = train_test_split(data_train, test_size=0.2, random_state=42)
        n_train = len(data_train)
        n_test = len(data_valid)

        st = 0
        y_pred = []
        for it in tqdm(range(n_batch)):
            if it % verbose_batch == 0:
                y_test_pred = [], []
                evaluate(sampler=self.sampler, environment=self.environment, data_test=data_test, train_samples=train_samples, epoch=-1, T=20, verbose=False)
                for i in range(0, n_test, batch_size):
                    data_batch = data_valid[i: min(i + batch_size, n_test)]
                    data_batch = self.read_train_batch(data_batch)
                    preds = self.Qa.run(data_batch, batch_size, train=False)
                    y_test_pred.extend(preds)

                test_eval = self.Qa.evaluate_metrics(y_test_pred)
                evals = [("test_" + u[0], u[1]) for u in test_eval]

                if it != 0:
                    train_eval = self.Qa.evaluate_metrics(y_pred)
                    train_eval = [("train_" + u[0], u[1]) for u in train_eval]
                    train_eval.extend(evals)
                    evals = train_eval
                print_result(evals, n_batch=it)
                y_pred = []

            if st == 0:
                np.random.shuffle(data_train)
            data_batch = data_train[st: min(st + batch_size, n_train)]
            data_batch = self.read_train_batch(data_batch)
            preds = self.Qa.run(data_batch, batch_size, train=True)
            y_pred.extend(preds)
            st = st + batch_size if st + batch_size < n_train else 0

    def read_train_batch(self, data_batch):
        data_state = [sparse_to_dense(self.train_samples[i][0][:j] + self.train_samples[i][0][j+1:], n_skill) for i, j in data_batch]
        data_skill = [self.train_samples[i][0][j] for i, j in data_batch]
        return data_state, data_skill


sample_lst, skill_cnt, jd_len = sample_info()
skill_p = [log(u * 1.0 / len(sample_lst)) for u in skill_cnt]
relation_lst = read_skill_graph()

if __name__ == "__main__":
    direct_name = "resume"
    env_params = {"lambda_d": 0.1, "beta": 0.2, 'pool_size': 100}

    sample_lst, skill_cnt, jd_len = sample_info()

    itemset = itemset_process(skill_cnt)
    d_estimator = DifficultyEstimator(item_sets=[u[0] for u in itemset], item_freq=[u[1] for u in itemset], n_samples=len(sample_lst))
    job_matcher = JobMatcher(n_top=100, skill_list=[u[0] for u in sample_lst], salary=[u[1] for u in sample_lst], w=5, th=10.0/9)
    environment = Environment(lambda_d=env_params['lambda_d'], d_estimator=d_estimator,
                              job_matcher=job_matcher, n_skill=n_skill)

    # train set
    train_samples = read_offline_samples(direct_name)
    train_samples = [(u[0], u[1]) for u in train_samples]
    data_train = read_pkl_data(HOME_PATH + "data/%s/train_test/traindata.pkl" % direct_name)
    lst = list(set([u[0] for u in data_train]))
    data_train = []
    for i in lst:
        data_train.extend([(i, j) for j in range(len(train_samples[i][0]))])
    N_train = len(data_train)
    print(N_train)

    # validation set
    data_test = read_pkl_data(HOME_PATH + "data/%s/train_test/validdata.pkl" % direct_name)

    # ----------------- Model ---------------------
    act_dnn = ActDNN(n_skill, verbose=1, lr=0.001, deep_layers=[128, 128, 128, 64, 16], dropout_deep=[0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                     activation=tf.nn.leaky_relu, random_seed=219, l2_reg=0.01)
    # act_dnn.load(HOME_PATH + "data/model/%s_ACT_DNN"%direct_name)

    # ----------------- Training ---------------------
    act_trainer = ActTrainer(act_dnn, train_samples=train_samples, environment=environment, pool_size=env_params['pool_size'])
    act_trainer.train(n_batch=10000, verbose_batch=500, batch_size=256, data_train=data_train)
    sampler = act_trainer.sampler

    evaluate(sampler=sampler, environment=environment, data_test=data_test, train_samples=train_samples, epoch=-1, T=20, verbose=False)

    # ----------------- 模型保存 -----------------------
    act_dnn.save(HOME_PATH + "data/model/%s_ACT_DNN" % direct_name)
