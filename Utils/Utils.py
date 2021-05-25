import numpy as np
import pickle


def print_result(data_eval, n_batch):
    print_str = "n_batch: %d, "%n_batch
    for i, name_val in enumerate(data_eval):
        if i != 0: print_str += ','
        print_str += "%s: %f" % name_val
    print(print_str)

def shuffle_in_unison_scary(data):
    rng_state = np.random.get_state()
    for lst in data:
        np.random.set_state(rng_state)
        np.random.shuffle(lst)


def dense_to_sparse(vis_list):
    lst = []
    for i, v in enumerate(vis_list):
        if v:
            lst.append(i)
    return lst


def sparse_to_dense(lst, n_s):
    vis_lst = [0] * n_s
    for i in lst:
        vis_lst[i] = 1
    return vis_lst


def read_pkl_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

