import pandas as pd
import pickle
import json
from CONFIG import JD_PATH, DICT_PATH, HOME_PATH, DICT_NAME_PATH, RESUME_PATH
from tqdm import tqdm
from Utils.Utils import read_pkl_data
import jieba


def skill_set_process(skill_set):
    '''
    :param skill_set: 技能集合，形如{level:理解，skill: 生命周期}
    :return: skill_list
    '''
    skill_set = json.loads(skill_set)
    skill_list = []
    for skill in skill_set:
        skill_list.append(skill_dict[skill])
    return skill_list


def _get_salary(salary_str):
    salary_str = salary_str.lower()
    num = None
    for i, _ in enumerate(salary_str):
        try:
            num = float(salary_str[: i + 1])
        except:
            break
    if num is None:
        return None
    if salary_str.find('k') != -1:
        num = num * 1000
    return num / 1000.0


def salary_process(salary_str):
    # 处理序列成数（K RMB）
    if salary_str.find('-') != -1:
        low, high = salary_str.split('-')
        low = _get_salary(low)
        high = _get_salary(high)
        return (low + high) / 2
    elif salary_str.find(u'以上') != -1:
        low = _get_salary(salary_str)
        return low
    elif salary_str.find(u'以下') != -1:
        high = _get_salary(salary_str)
        return high
    return 0


def itemset_process(skill_cnt):
    with open(HOME_PATH + "data/item_set.pkl", "rb") as f:
        data = pickle.load(f)
    T = []
    for itemset, frequency in tqdm(data):
        if len(itemset) < 2: continue
        itemset.sort(key=lambda x: skill_cnt[x], reverse=True) # 出现次数的降序
        T.append((itemset, frequency))

    for s in range(n_skill):
        T.append(([s], skill_cnt[s]))
    return T


def read_skill_graph(thres=0.1):
    with open(HOME_PATH + "data/skill_graph.pkl", 'rb') as f:
        skill_G = pickle.load(f, encoding='bytes')
    # return skill_G
    x = skill_G.nonzero()
    val = skill_G[x].tolist()[0]
    x, y = x
    print(len(x), len(y), len(val))
    df = pd.DataFrame()
    df['a1'] = x
    df['a2'] = y
    df['val'] = val
    df = df[df['val'] > thres]
    df = df.drop('val', axis=1)
    a1_lst = df['a1'].tolist() + df['a2'].tolist()
    a2_lst = df['a2'].tolist() + df['a1'].tolist()
    df = pd.DataFrame()
    df['a1'] = a1_lst
    df['a2'] = a2_lst
    df = df.drop_duplicates()
    #print(df.shape[0])
    lst = [[] for u in range(n_skill)]
    for a1, a2 in df[['a1', 'a2']].values.tolist():
        lst[a1].append(a2)
    #print([len(u) for u in lst])
    return lst


def read_jd():
    '''
    :param path: jd_expanded.csv存放地址
    :param dict_path: 技能编号存放地址
    :return:  sample_lst: list(list, salary) 形式，每个sample是技能list和对应的薪酬
    '''
    path = JD_PATH
    df = pd.read_csv(path)[['id', 'skill_set', 'job_salary']].sort_values(by='id', ascending=True)
    n_sample = df.shape[0]
    assert n_sample == df['id'].max() + 1, "工作发布的编号有错误，id最大值应等于样本数-1"
    sample_lst = []

    for id, skill_set, job_salary in tqdm(df.values.tolist()):
        skill_list = skill_set_process(skill_set)
        salary = salary_process(job_salary)
        sample_lst.append((skill_list, salary))
    return sample_lst


def read_dict(): # 要改，以及config要改
    df = pd.read_csv(DICT_PATH)[['skill', 'id']]
    skill_dict = {}
    for word, id in df.values.tolist():
        skill_dict[word] = id
    n_skill = max(skill_dict.values()) + 1
    return skill_dict, n_skill


def read_skill_name(skill_dict):
    df = pd.read_csv(DICT_NAME_PATH)[['word', 'merge']]
    df = df[df['merge'].apply(lambda x: x in skill_dict)]
    skill_name_dict = {}
    for word, id in df.values.tolist():
        skill_name_dict[word] = id
    return skill_name_dict


def sample_info():
    sample_lst = read_jd()
    skill_cnt = [1] * n_skill
    l = []
    for i, sample in enumerate(sample_lst):
        skill_lst, salary = sample
        for skill in skill_lst:
            skill_cnt[skill] += 1
        l.append(len(skill_lst))
    return sample_lst, skill_cnt, l


def json2skill_list(jd_str, skill_cnt):
    skill_list = [skill_dict[u] for u in json.loads(jd_str)]
    skill_list.sort(key=lambda x: skill_cnt[x], reverse=True)  # 大到小
    return skill_list


def read_resume_list(skill_cnt=None):
    if skill_cnt is None:
        _, skill_cnt, _ = sample_info()
    series = pd.read_csv(RESUME_PATH)['skill_set'].apply(lambda x: json2skill_list(x, skill_cnt))
    return series.tolist()


def reverse_dict():
    lst = []
    for skill, value in skill_dict.items():
        lst.append((skill, value))
    lst.sort(key=lambda x: x[1], reverse=False)
    ret_lst = []
    for skill, value in lst:
        while value != len(ret_lst):
            ret_lst.append('')
        ret_lst.append(skill)
    return ret_lst


def read_offline_samples(direct_name):
    return read_pkl_data(HOME_PATH + "data/%s/off_data/train_set.pkl" % direct_name)


skill_dict, n_skill = read_dict()
skill_lst = reverse_dict()
skill_name_dict = read_skill_name(skill_dict)
