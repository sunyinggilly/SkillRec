import random
from Utils.JobReader import skill_lst
from math import log

def get_act_pool(state, n_a, relational_lst, pool_size):
    cnt = [0] * n_a
    for i in range(n_a):
        if state[i] == 1:
            for u in relational_lst[i]:
                cnt[u] += 1
    skill_id = list(range(n_a))
    skill_id.sort(key=lambda x: cnt[x] + 1e-5 * x, reverse=True)
    ret = []
    for u in range(n_a):
        s = skill_id[u]
        if state[s] == 0:
            ret.append(s)
        if len(ret) == pool_size:
            break
    return ret

class DistributionPoolSampler(object):
    def __init__(self, p, n_a, pool_size, relational_lst):
        self.n_a = n_a
        self.pool_size = pool_size
        self.relational_lst = relational_lst
        self.p = p

    def judge(self, p, prob, x):
        return prob < p[x]

    def binary_search(self, p, prob, l, r):
        if r - l <= 1:
            if self.judge(p, prob, l): return l
            return r
        m = (l + r) // 2
        if self.judge(p, prob, m): return self.binary_search(p, prob, l, m)
        return self.binary_search(p, prob, m + 1, r)

    def sample(self, state):

        pool = get_act_pool(state, self.n_a, self.relational_lst, self.pool_size)
        p = [self.p[u] for u in pool]
        for i in range(len(p) - 1):
            p[i + 1] += p[i]

        s = pool[self.binary_search(p, random.random() * p[-1], 0, self.pool_size - 1)]
        while state[s] != 0:
            s = pool[self.binary_search(p, random.random() * p[-1], 0, self.pool_size - 1)]
        return s, None

class EpsilonGreedySampler(object):
    def __init__(self, Qa, epsilon, n_a):
        self.epsilon = epsilon
        self.n_a = n_a
        self.Qa = Qa

    def sample(self, state, retQ=False):
        if random.random() > self.epsilon:
            s_ret = random.randint(0, self.n_a - 1)
            while state[s_ret] != 0:
                s_ret = random.randint(0, self.n_a - 1)
            if retQ:
                q_ret = self.Qa.estimate_single(state, s_ret)
            else:
                q_ret = None
        else:
            q_ret, s_ret = self.Qa.estimate_maxq_action(state)
        return s_ret, q_ret

    def sample_batch(self, state, retQ=False):
        q_ret, s_ret = self.Qa.estimate_maxq_batch(state)
        q_ret_batch, s_ret_batch = [], []
        for state_now, q, s in zip(state, q_ret, s_ret):
            if random.random() > self.epsilon:
                s = random.randint(0, self.n_a - 1)
                while state_now[s] != 0:
                    s = random.randint(0, self.n_a - 1)
                if retQ:
                    q = self.Qa.estimate_single(state_now, s)
                else:
                    q = None
            q_ret_batch.append(q)
            s_ret_batch.append(s)
        return s_ret_batch, q_ret_batch

class EpsilonGreedyPoolSampler(object):
    def __init__(self, relational_lst, Qa, epsilon, n_a, pool_size):
        self.epsilon = epsilon
        self.n_a = n_a
        self.Qa = Qa
        self.relational_lst = relational_lst
        self.pool_size = pool_size

    def judge(self, p, prob, x):
        return prob < p[x]

    def binary_search(self, p, prob, l, r):
        if r - l <= 1:
            if self.judge(p, prob, l): return l
            return r
        m = (l + r) // 2
        if self.judge(p, prob, m): return self.binary_search(p, prob, l, m)
        return self.binary_search(p, prob, m + 1, r)

    def sample(self, state, pool=None, retQ=False):
        if pool is None:
            pool = get_act_pool(state, self.n_a, self.relational_lst, self.pool_size)
        if random.random() > self.epsilon:
            s_ret = pool[random.randint(0, len(pool) - 1)]
            while state[s_ret] != 0:
                s_ret = pool[random.randint(0, len(pool) - 1)]
            if retQ:
                q_ret = self.Qa.estimate_single(state, s_ret)
            else:
                q_ret = None
        else:
            q_ret, s_ret = self.Qa.estimate_maxq_action(state, pool)
        return s_ret, q_ret

class BestStrategyPoolSampler(object):
    def __init__(self, relational_lst, Qa, n_a, pool_size):
        self.n_a = n_a
        self.Qa = Qa
        self.relational_lst = relational_lst
        self.pool_size = pool_size

    def sample(self, state, pool=None):
        if pool is None:
            pool = get_act_pool(state, self.n_a, self.relational_lst, self.pool_size)
        q_ret, s_ret = self.Qa.estimate_maxq_action(state, pool)
        return s_ret, q_ret

class BestStrategySampler(object):
    def __init__(self, Qa, n_a):
        self.n_a = n_a
        self.Qa = Qa

    def sample(self, state):
        q_ret, s_ret = self.Qa.estimate_maxq_action(state)
        return s_ret, q_ret

    def sample_batch(self, state):
        q_ret, s_ret = self.Qa.estimate_maxq_batch(state)
        return s_ret, q_ret


class DistributionSampler(object):
    def __init__(self, p, n_a):
        self.n_a = n_a
        self.p = p
        for i in range(len(p) - 1):
            self.p[i + 1] += self.p[i]

    def judge(self, prob, x):
        return prob < self.p[x]

    def binary_search(self, prob, l, r):
        if r - l <= 1:
            if self.judge(prob, l): return l
            return r
        m = (l + r) // 2
        if self.judge(prob, m): return self.binary_search(prob, l, m)
        return self.binary_search(prob, m + 1, r)

    def sample(self, state):
        s = self.binary_search(random.random(), 0, self.n_a - 1)
        while state[s] != 0:
            s = self.binary_search(random.random(), 0, self.n_a - 1)
        return s, None


class GreedySampler(object):
    def __init__(self, relational_lst, environment, n_a, pool_size, rtype="salary"):
        self.n_a = n_a
        self.environment = environment
        self.jm = environment.job_matcher
        self.de = environment.d_estimator
        self.relational_lst = relational_lst
        self.pool_size = pool_size
        self.type = rtype

    def sample(self, state):
        pool = get_act_pool(state, self.n_a, self.relational_lst, self.pool_size)
        if self.type == 'salary':
            r_lst = [self.jm.predict_salary(s) for s in pool]
        elif self.type == 'easy':
            r_lst = [self.de.predict_easy(s) for s in pool]
        else:
            r_lst = [self.environment.get_reward(salary=self.jm.predict_salary(s), easy=self.de.predict_easy(s)) for s in pool]

        s_ret, r_ret = -1, -1000
        for s, rnow in zip(pool, r_lst):
            if rnow > r_ret:
                s_ret = s
                r_ret = rnow
        return s_ret, r_ret

class GreedyUnionSampler(object):
    def __init__(self, relational_lst, environment, Qa, n_a, pool_size, beta):
        self.n_a = n_a
        self.environment = environment
        self.jm = environment.job_matcher
        self.de = environment.d_estimator
        self.relational_lst = relational_lst
        self.pool_size = pool_size
        self.Qa = Qa
        self.beta = beta

    def sample(self, state):
        pool = get_act_pool(state, self.n_a, self.relational_lst, self.pool_size)
        r_lst = [self.environment.get_reward(salary=self.jm.predict_salary(s), easy=self.de.predict_easy(s)) for s in pool]
        s_ret, r_ret = -1, -1000
        state_lst = []
        pool_lst = []
        for s in pool:
            state_now = state.copy()
            state_now[s] = 1
            state_lst.append(state_now)
            pool_lst.append(get_act_pool(state_now, self.n_a, self.relational_lst, self.pool_size))

        q_ret, _ = self.Qa.estimate_maxq_batch(state_lst, pool_lst)
        salary_ret, easy_ret = q_ret
#        for s, r, sal, ease in zip(pool, r_lst, salary_ret, easy_ret):
#            print(s, r, sal, ease)
        q_ret = [rnow + (1 - self.beta) * self.environment.get_reward(salary=sal, easy=easy) for rnow, sal, easy in zip(r_lst, salary_ret, easy_ret)]

        for s, rnow in zip(pool, q_ret):
            if rnow > r_ret:
                s_ret = s
                r_ret = rnow
#        print(s_ret)
        return s_ret, r_ret

    def sample_pre(self, state):
        pool = get_act_pool(state, self.n_a, self.relational_lst, self.pool_size)
        r_lst = [self.environment.get_reward(salary=self.jm.predict_salary(s), easy=self.de.predict_easy(s)) for s in pool]
        s_ret, r_ret = -1, -1000
        for s, rnow in zip(pool, r_lst):
            state[s] = 1
            pool_nxt = get_act_pool(state, self.n_a, self.relational_lst, self.pool_size)
            q_ret, _ = self.Qa.estimate_maxq_action(state, pool_nxt)
            q_ret = self.environment.get_reward(salary=q_ret[0], easy=q_ret[1])
            rnow = rnow + (1-self.beta) * q_ret
            state[s] = 0
            if rnow > r_ret:
                s_ret = s
                r_ret = rnow
        return s_ret, r_ret

