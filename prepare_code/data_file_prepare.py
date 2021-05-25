import sys
sys.path.append("../")
import pickle
from tqdm import tqdm
from random import randint
from Utils.Utils import read_pkl_data
from random import shuffle

from CONFIG import HOME_PATH
from Environment.DifficultyEstimatorGLinux import DifficultyEstimator
from Environment.Environment import Environment
from Environment.JobMatcherLinux import JobMatcher
from Sampler import DistributionSampler
from Utils.JobReader import sample_info, n_skill, skill_lst, read_resume_list, read_offline_samples, read_skill_graph, itemset_process


def create_cinput_skill_list():
    for skill_lst, salary in sample_lst:
        skill_lst.sort(key=lambda x: skill_cnt[x], reverse=False)
    return sample_lst


def create_data_with_neg(sequence_list, direct_name):
    p = [1.0 / n_skill] * n_skill
    sampler = DistributionSampler(p, n_skill)
    data, negr_lst, r_lst = [], [], []
    for lst_id, skill_lst in enumerate(sequence_list[297:]):
        environment.clear()
        skill_lst.sort(key=lambda x: skill_cnt[x], reverse=True) # 大到小
        neg_skills = []
        
        vis = [0] * n_skill
        vis[skill_lst[0]] = 1
        environment.add_skill(skill_lst[0])
        for skill in skill_lst[1:]:
            s, _ = sampler.sample(vis)
            easy_neg = environment.d_estimator.predict_easy(s)
            salary_neg = environment.job_matcher.predict_salary(s)
            easy_pos, salary_pos, _ = environment.add_skill(skill, evaluate=True)

            vis[skill] = 1
            neg_skills.append(s)
            negr_lst.append((easy_neg, salary_neg))
            r_lst.append((easy_pos, salary_pos))
        data.append((skill_lst, r_lst, neg_skills, negr_lst))

    with open(HOME_PATH + "data/%s/off_data/train_set.pkl"%direct_name, 'wb') as f:
        pickle.dump(data, f)
    return


def validation_split(direct_name, N_test=5000):
    train_samples = read_offline_samples(direct_name)  # skill_lst, r_lst[i]
    sample_id = list(range(len(train_samples)))
    N_data = len(sample_id)
    N_train = N_data - N_test
    shuffle(sample_id)
    train_id = sample_id[N_test:]
    test_id = sample_id[: N_test]

    # 训练集 换一种做法： 训练集和测试集按比例划分，然后分别扩展。 接下来测试集里随便挑500个做validation。
    data_train = []
    for i in train_id: # skill_lst, r_lst
        dt = train_samples[i]
        m = len(dt[0])
        data_train.append((i, m - 1))

    with open(HOME_PATH + "data/%s/train_test/traindata.pkl"%direct_name, 'wb') as f:
        pickle.dump(data_train, f)

    # 测试集
    data_test = []
    for i in test_id: # skill_lst, r_lst
        dt = train_samples[i]
        m = len(dt[0])
        data_test.append((i, m-1))

    with open(HOME_PATH + "data/%s/train_test/testdata.pkl"%(direct_name), 'wb') as f:
        pickle.dump(data_test, f)
    print(len(data_test))
    data_valid = []
    for i in range(500):
        x = randint(0, len(data_test) - 1)
        data_valid.append(data_test[x])

    with open(HOME_PATH + "data/%s/train_test/validdata.pkl"%(direct_name), 'wb') as f:
        pickle.dump(data_valid, f)


if __name__ == "__main__":
    env_params = {"lambda_d": 0.1, "beta": 0.2, 'pool_size': 50}

    sample_lst, skill_cnt, jd_len = sample_info()
    n_jd = len(sample_lst)
    resume_list = read_resume_list(skill_cnt)

    itemset = itemset_process(skill_cnt)
    d_estimator = DifficultyEstimator(item_sets=[u[0] for u in itemset], item_freq=[u[1] for u in itemset], n_samples=len(sample_lst))
    job_matcher = JobMatcher(n_top=100, skill_list=[u[0] for u in sample_lst], salary=[u[1] for u in sample_lst], w=5, th=10.0 / 9)

    environment = Environment(lambda_d=env_params['lambda_d'], d_estimator=d_estimator, job_matcher=job_matcher, n_skill=n_skill)

    create_data_with_neg(resume_list, "resume")
    validation_split("resume")
