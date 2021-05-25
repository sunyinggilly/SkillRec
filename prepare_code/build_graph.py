# -*- coding: utf-8 -*
import pandas as pd
import pickle
from scipy.sparse import coo_matrix
from CONFIG import HOME_PATH
from Utils.JobReader import read_jd, n_skill


def min_max(series):
    mx = series.max()
    mi = series.min()
    return series.apply(lambda x: (x * 1.0 - mi) / (mx - mi))


if __name__ == "__main__":
    sample_lst = read_jd()

    datalst = []
    vallst = []
    for skill_set, salary in sample_lst:
        len_now = len(skill_set)
        for i in range(len_now):
            vallst.append([skill_set[i]])
            for j in range(i + 1, len_now):
                datalst.append((skill_set[i], skill_set[j]))
                datalst.append((skill_set[j], skill_set[i]))

    df_mat = pd.DataFrame(datalst, columns=['node_1', 'node_2'])
    df_mat['count'] = 1
    df_mat = df_mat.groupby(['node_1', 'node_2']).sum().reset_index()

    df_zero = pd.DataFrame([(ind, 0) for ind in range(n_skill)], columns=['node_1', 'all_count'])

    df_value = pd.DataFrame(vallst, columns=['node_1'])
    df_value['all_count'] = 1
    df_value = df_value.append(df_zero)
    df_value = df_value.groupby('node_1').sum().reset_index().sort_values(by='node_1')

    df_merge = pd.merge(df_mat, df_value, on='node_1')
    df_merge['count'] = df_merge[['count', 'all_count']].apply(lambda x: 1.0 * x[0] / x[1], axis=1)
    df_mat = df_merge[['node_1', 'node_2', 'count']]
    G = coo_matrix((df_mat['count'].tolist(), (df_mat['node_1'].tolist(), df_mat['node_2'].tolist())), shape=(n_skill, n_skill)).todense()

    with open(HOME_PATH + "data/skill_graph.pkl", 'wb') as f:
        pickle.dump(G, f)
