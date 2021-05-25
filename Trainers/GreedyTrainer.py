import sys
sys.path.append("../../")
sys.path.append("../")
from Environment.JobMatcherLinux import JobMatcher
from Utils.JobReader import n_skill, sample_info, read_offline_samples, read_skill_graph, itemset_process
from Environment.DifficultyEstimatorGLinux import DifficultyEstimator
from Models.DNN import DNN
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import tensorflow as tf
from Utils.Utils import sparse_to_dense, read_pkl_data, print_result
from Utils.Functions import evaluate
from Sampler import BestStrategyPoolSampler
from CONFIG import HOME_PATH
from tqdm import tqdm
from Environment.Environment import Environment
from sklearn.model_selection import train_test_split
import numpy as np
from math import log


class ShortTrainer(object):
    def __init__(self, Qa, environment, train_samples, pool_size, md):
        self.environment = environment
        self.sampler = BestStrategyPoolSampler(relation_lst, Qa, n_skill, pool_size)
        self.Qa = Qa
        self.train_samples = train_samples
        self.md = md
        self.pool_size = pool_size

    def train(self, n_batch, verbose_batch, batch_size, data_train, data_test=None):
        data_train, data_valid = train_test_split(data_train, test_size=0.2, random_state=42)
        n_train = len(data_train)
        n_test = len(data_valid)

        st = 0
        y_train_true, y_train_pred = [], []
        for it in tqdm(range(n_batch)):
            if it % verbose_batch == 0:
                if data_test is not None:
                    evaluate(self.sampler, self.environment, data_test, self.train_samples, it / verbose_batch, T=26)
                y_test_true, y_test_pred = [], []
                for i in range(0, n_test, batch_size):
                    data_batch = data_valid[i: min(i + batch_size, n_test)]
                    data_batch = self.read_train_batch(data_batch, md=self.md)
                    predictions = self.Qa.run(data_batch, batch_size, train=False)

                    y_test_pred.extend(predictions)
                    y_test_true.extend(data_batch[2])

                test_eval = self.Qa.evaluate_metrics(y_test_pred, y_test_true)
                evals = [("test_" + u[0], u[1]) for u in test_eval]
                if it != 0:
                    train_eval = self.Qa.evaluate_metrics(y_train_pred, y_train_true)
                    train_eval = [("train_" + u[0], u[1]) for u in train_eval]
                    train_eval.extend(evals)
                    evals = train_eval
                print_result(evals, n_batch=it)

                y_train_true, y_train_pred = [], []

            if st == 0:
                np.random.shuffle(data_train)
            data_batch = data_train[st: min(st + batch_size, n_train)]
            data_batch = self.read_train_batch(data_batch, md=self.md)
            predictions = self.Qa.run(data_batch, batch_size, train=True)

            y_train_pred.extend(predictions)
            y_train_true.extend(data_batch[2])

            st = st + batch_size if st + batch_size < n_train else 0

    def read_train_batch(self, data_batch, md):
        data_state = [sparse_to_dense(self.train_samples[i][0][:j], n_skill) for i, j in data_batch]
        data_skill = [self.train_samples[i][0][j] for i, j in data_batch]
        data_easy = [self.train_samples[i][1][j - 1][0] for i, j in data_batch]
        data_salary = [self.train_samples[i][1][j - 1][1] for i, j in data_batch]

        data_state_neg = data_state.copy()
        data_skill_neg = [self.train_samples[i][2][j - 1] for i, j in data_batch]
        data_easy_neg = [self.train_samples[i][3][j - 1][0] for i, j in data_batch]
        data_salary_neg = [self.train_samples[i][3][j - 1][1] for i, j in data_batch]

        data_state.extend(data_state_neg)
        data_skill.extend(data_skill_neg)
        data_easy.extend(data_easy_neg)
        data_salary.extend(data_salary_neg)
        data_r = [self.environment.get_reward(easy=easy, salary=salary) for easy, salary in zip(data_easy, data_salary)]

        if md == 'salary':
            return data_state, data_skill, data_salary
        elif md == 'easy':
            return data_state, data_skill, data_easy
        else:
            return data_state, data_skill, data_r


sample_lst, skill_cnt, jd_len = sample_info()
skill_p = [log(u * 1.0 / len(sample_lst)) for u in skill_cnt]
relation_lst = read_skill_graph()

if __name__ == "__main__":
    direct_name = "resume"
    md = sys.argv[1]
    env_params = {"lambda_d": 0.1, "beta": 0.2, 'pool_size': 100}
    on_policy_params = {"n_memory": 100000}

    itemset = itemset_process(skill_cnt)
    d_estimator = DifficultyEstimator(item_sets=[u[0] for u in itemset], item_freq=[u[1] for u in itemset], n_samples=len(sample_lst))
    job_matcher = JobMatcher(n_top=100, skill_list=[u[0] for u in sample_lst], salary=[u[1] for u in sample_lst], w=5, th=10.0 / 9)

    environment = Environment(lambda_d=env_params['lambda_d'], d_estimator=d_estimator,
                              job_matcher=job_matcher, n_skill=n_skill)

    train_samples = read_offline_samples(direct_name)

    # 训练集
    data_train = read_pkl_data(HOME_PATH + "data/%s/train_test/traindata.pkl" % direct_name)
    lst = list(set([u[0] for u in data_train]))
    data_train = []
    for i in lst:
        data_train.extend([(i, j) for j in range(len(train_samples[i][0]))])
    N_train = len(data_train)
    print(N_train)

    # 测试集
    data_valid = read_pkl_data(HOME_PATH + "data/%s/train_test/validdata.pkl" % direct_name)

    # ----------------- 初始模型 ---------------------
    Qa = DNN(n_skill, verbose=1, lr=0.001,
             deep_layers=[128, 128, 128, 64, 16], dropout_deep=[0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
             activation=tf.nn.leaky_relu, random_seed=2019, l2_reg=0.01)

    # ----------------- 模型训练 ---------------------
    short_trainer = ShortTrainer(Qa, train_samples=train_samples, environment=environment, pool_size=env_params['pool_size'])

    short_trainer.train(n_batch=40000, verbose_batch=10000, batch_size=256, data_train=data_train, data_test=None, md=md)
    sampler = short_trainer.sampler
    data_test = read_pkl_data(HOME_PATH + "data/%s/train_test/testdata.pkl" % direct_name)
    evaluate(sampler=sampler, environment=environment, data_test=data_test, train_samples=train_samples, epoch=-1, T=20, verbose=False)

    # ----------------- 模型保存 -----------------------
    Qa.save(HOME_PATH + "data/model/%s_DNN_%s" % (direct_name, md))
