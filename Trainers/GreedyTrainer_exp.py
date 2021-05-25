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
from Utils.Utils import sparse_to_dense, read_pkl_data
from Utils.Functions import evaluate
from Sampler import BestStrategyPoolSampler, EpsilonGreedyPoolSampler
from CONFIG import HOME_PATH
from tqdm import tqdm
from Environment.Environment import Environment
from Trainers.Memory import Memory2
from random import randint
from math import log


class OnPolicyTrainer(object):

    def __init__(self, Qnet, environment, train_samples, relational_lst, memory, pool_size, md):
        self.environment = environment
        self.relational_lst = relational_lst
        self.pool_size = pool_size

        self.sampler = EpsilonGreedyPoolSampler(relation_lst, Qnet, 0.7, n_skill, pool_size=pool_size)
        self.best_sampler = BestStrategyPoolSampler(relational_lst, Qnet, n_skill, pool_size)

        self.Qnet = Qnet
        self.train_samples = train_samples

        self.memory = memory
        self.md = md

    def train(self, n_batch, batch_size, data_train, data_valid, save_path, verbose_batch=500, T=26):
        n_data = len(data_train)
        for it in tqdm(range(n_batch)):
            if it > 0 and it % verbose_batch == 0:
                evaluate(self.best_sampler, self.environment, data_valid, self.train_samples, it / verbose_batch, T=T, verbose=True)
                self.sampler.epsilon += (1 - self.sampler.epsilon) * 0.1
                self.Qnet.save(save_path)

            self.environment.clear()

            # 采样一个前缀
            k = randint(0, n_data - 1)
            x, y = data_train[k]
            prefix = self.train_samples[x][0]

            salary_pre = self.environment.add_prefix(prefix)
            for t in range(T):
                state_pre = self.environment.state_list.copy()
                s, _ = self.sampler.sample(self.environment.state)
                easy, salary, r = self.environment.add_skill(s, evaluate=True)
                self.memory.store((state_pre, s, (easy, salary - salary_pre)))  # 要有遗忘才行
                salary_pre = salary

            if self.memory.get_size() > batch_size * 10:
                data_batch = self.memory.sample(batch_size)
                data_batch = self.transform_train_batch(data_batch)
                self.Qnet.run(data_batch, batch_size, train=True)
        return

    def transform_train_batch(self, data_batch):
        # 输入一个batch的数据，形如[(i, j)]，返回data_state, data_skill, data_q
        data_state = [sparse_to_dense(u[0], n_skill) for u in data_batch]
        data_skill = [u[1] for u in data_batch]
        # print(data_skill)
        data_easy = [u[2][0] for u in data_batch]
        data_salary = [u[2][1] for u in data_batch]
        data_r = [self.environment.get_reward(easy=easy, salary=salary) for easy, salary in zip(data_easy, data_salary)]

        if self.md == 'salary':
            return data_state, data_skill, data_salary
        elif self.md == 'easy':
            return data_state, data_skill, data_easy
        else:
            return data_state, data_skill, data_r


# 参数读取
sample_lst, skill_cnt, jd_len = sample_info()
skill_p = [log(u * 1.0 / len(sample_lst)) for u in skill_cnt]
relation_lst = read_skill_graph()

if __name__ == "__main__":
    direct_name = "resume"
    md = sys.argv[1]
#    env_params = {"lambda_d": 0.1, "beta": 0.2, 'pool_size': 50}
    env_params = {"lambda_d": 0.1, "beta": 0.2, 'pool_size': 100}
    # 难度 & 奖励
    itemset = itemset_process(skill_cnt)
    d_estimator = DifficultyEstimator(item_sets=[u[0] for u in itemset], item_freq=[u[1] for u in itemset], n_samples=len(sample_lst))
    job_matcher = JobMatcher(n_top=100, skill_list=[u[0] for u in sample_lst], salary=[u[1] for u in sample_lst], w=5, th=10.0 / 9)

    environment = Environment(lambda_d=env_params['lambda_d'], d_estimator=d_estimator,
                              job_matcher=job_matcher, n_skill=n_skill)

    train_samples = read_offline_samples(direct_name)

    # 记忆单元
    memory = Memory2(20000)

    # 训练集
    data_train = read_pkl_data(HOME_PATH + "data/%s/train_test/traindata.pkl" % direct_name)
    N_train = len(data_train)

    # 测试集
    data_valid = read_pkl_data(HOME_PATH + "data/%s/train_test/validdata.pkl" % direct_name)

    # ----------------- 初始模型 ---------------------
    Qa = DNN(n_skill, verbose=1, lr=0.001,
             deep_layers=[256, 256, 256, 256, 256], dropout_deep=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
             activation=tf.nn.leaky_relu, random_seed=2019, l2_reg=0.01)

    # ----------------- 模型读取 ---------------------
    # Qa.load(HOME_PATH + "data/model/%s_DNN_%s_exp" % (direct_name, md))

    # ----------------- 模型训练 ---------------------
    on_trainer = OnPolicyTrainer(Qa, environment=environment, train_samples=train_samples,
                                 memory=memory, relational_lst=relation_lst, pool_size=env_params['pool_size'], md=md)
    on_trainer.train(n_batch=480000, batch_size=64, data_train=data_train, data_valid=data_valid, verbose_batch=10240,
                     save_path=HOME_PATH + "data/model/%s_DNN_%s_exp" % (direct_name, md), T=20)
    sampler = BestStrategyPoolSampler(relation_lst, Qa, n_skill, pool_size=env_params['pool_size'])

    data_test = read_pkl_data(HOME_PATH + "data/%s/train_test/testdata.pkl" % direct_name)
    evaluate(sampler=sampler, environment=environment, data_test=data_valid,
             train_samples=train_samples, epoch=-1, T=20, verbose=False)

    # ----------------- 模型保存 -----------------------
    Qa.save(HOME_PATH + "data/model/%s_DNN_%s_exp" % (direct_name, md))
