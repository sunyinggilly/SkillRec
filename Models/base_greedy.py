import sys
sys.path.append("../../")
sys.path.append("../")
from Environment.JobMatcherLinux import JobMatcher
from Utils.JobReader import n_skill, sample_info, read_offline_samples, read_skill_graph, itemset_process
from Environment.DifficultyEstimatorGLinux import DifficultyEstimator
from Utils.Utils import read_pkl_data
from Utils.Functions import evaluate
from Sampler import GreedySampler
from CONFIG import HOME_PATH
from Environment.Environment import Environment
from math import log


# 参数读取
sample_lst, skill_cnt, jd_len = sample_info()
skill_p = [log(u * 1.0 / len(sample_lst)) for u in skill_cnt]
relation_lst = read_skill_graph()

if __name__ == "__main__":
    direct_name = "resume"

    env_params = {"lambda_d": float(sys.argv[1]), "beta": 0.1, 'pool_size': int(sys.argv[2])}

    # 难度 & 奖励
    itemset = itemset_process(skill_cnt)
    d_estimator = DifficultyEstimator(item_sets=[u[0] for u in itemset], item_freq=[u[1] for u in itemset], n_samples=len(sample_lst))
    job_matcher = JobMatcher(n_top=100, skill_list=[u[0] for u in sample_lst], salary=[u[1] for u in sample_lst], w=5, th=10.0 / 9)

    environment = Environment(lambda_d=env_params['lambda_d'], d_estimator=d_estimator, job_matcher=job_matcher, n_skill=n_skill)

    train_samples = read_offline_samples(direct_name)

    # 测试集
    data_valid = read_pkl_data(HOME_PATH + "data/%s/train_test/validdata.pkl" % direct_name)
    data_test = read_pkl_data(HOME_PATH + "data/%s/train_test/testdata.pkl" % direct_name)

    # ----------------- 模型训练 ---------------------
    sampler = GreedySampler(relation_lst, environment, n_skill, pool_size=env_params['pool_size'], rtype='salary')
    evaluate(sampler=sampler, environment=environment, data_test=data_test, train_samples=train_samples, epoch=-1, T=20, verbose=False)

    sampler = GreedySampler(relation_lst, environment, n_skill, pool_size=env_params['pool_size'], rtype='easy')
    evaluate(sampler=sampler, environment=environment, data_test=data_test, train_samples=train_samples, epoch=-1, T=20, verbose=False)

    sampler = GreedySampler(relation_lst, environment, n_skill, pool_size=env_params['pool_size'], rtype='reward')
    evaluate(sampler=sampler, environment=environment, data_test=data_test, train_samples=train_samples, epoch=-1, T=20, verbose=False)
