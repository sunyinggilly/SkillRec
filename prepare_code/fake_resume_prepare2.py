import pandas as pd
from CONFIG import HOME_PATH
from Utils.JobReader import sample_info, skill_dict, n_skill, skill_lst
import json
from tqdm import tqdm
import random

skill_count_lst = [0] * n_skill

_, skill_cnt, _ = sample_info()
df = pd.read_csv(HOME_PATH + "data/jd_filter_2.csv")
print(df.shape[0])
df = df.drop_duplicates(subset='skill_set')
df['skill_set'] = df['skill_set'].apply(lambda x: [skill_dict[u] for u in json.loads(x)])
df = df[df['skill_set'].apply(lambda x: len(x) < 40)]

sz = 100
df_use = pd.DataFrame()
for i in tqdm(range(0, 150000, sz)):
    df['val'] = df['skill_set'].apply(lambda x: sum([skill_count_lst[u] for u in x]) * 1.0 / len(x))
    df = df.sort_values(by='val', ascending=True)
    df_random = df.head(50000).sample(sz)
    df_use = df_use.append(df_random, ignore_index=True)

    lsts = df_random['skill_set']
    for lst in lsts:
        for u in lst:
            skill_count_lst[u] += 1

print(df_use['skill_set'])
df_use['skill_set'] = df_use['skill_set'].apply(lambda x: json.dumps([skill_lst[u] for u in x], ensure_ascii=False))
df_use.drop('val', axis=1).to_csv(HOME_PATH + "data/fake_resume_balance.csv", index=False)
