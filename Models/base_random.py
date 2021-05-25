from Sampler import DistributionSampler, BestSalarySampler, BestShortSampler
from Utils.JobReader import n_skill, sample_info, read_offline_samples
from Utils.Utils import read_pkl_data
from Utils.Functions import evaluate
from Environment.CorrelationDifficulty.DifficultyEstimator import get_estimator
from Environment.Environment import Environment
from Environment.JobMatcher import JobMatcher
from CONFIG import HOME_PATH

if __name__ == "__main__":
    env_params = {"lambda_d": 1}
    mode = "uniform"
    T = 26
    sample_lst, skill_cnt, _ = sample_info()

    # 难度 & 奖励
    d_estimator = get_estimator(skill_cnt, len(sample_lst))
    job_matcher = JobMatcher(n_top=30, sample_lst=sample_lst, cnt=skill_cnt)
    environment = Environment(env_params['lambda_d'], d_estimator, job_matcher, n_skill)

    if mode == 'uniform':
        # 均匀分布
        uniform_p = [1.0 / n_skill] * n_skill
        sampler = DistributionSampler(p=uniform_p, n_a=n_skill)
    elif mode == 'frequency':
        cnt_sum = sum(skill_cnt)
        freq_p = [u * 1.0 / cnt_sum for u in skill_cnt]
        sampler = DistributionSampler(p=freq_p, n_a=n_skill)
    elif mode == 'salary':
        sampler = BestSalarySampler(job_matcher, n_skill, skill_cnt)
    else:
        sampler = BestShortSampler(environment, n_skill, skill_cnt) # 太慢了，换一种方式

    train_samples = read_offline_samples("jd") # skill_lst, r_lst

    N_test = 500
    for it in range(1):
        data_test = read_pkl_data(HOME_PATH + "data/jd/train_test/testdata_%d_%d.pkl" % (N_test, it))
        evaluate(sampler, environment, data_test, train_samples, it, T=T)
