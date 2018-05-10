#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/10 下午10:19
# @Author  : Shiloh Leung
# @Site    : 
# @File    : experiment_tool.py
# @Software: PyCharm
import numpy as np


def index_generator(tensor):
    return tuple([np.random.randint(dim) for dim in tensor.shape])


def split_train_test(tensor, rate=0.1, balanced=False):
    num_test = int(np.sum(tensor != 0) * rate)

    if balanced is False:
        p = rate / (1 - np.mean(tensor == 0))
        test_mask = np.random.choice([True, False], size=tensor.shape, p=[p, 1 - p])
        test_mask[tensor == 0] = False
    else:
        half_num_test = int(num_test / 2)
        assert np.sum(tensor == 1) > half_num_test / 2, "test set should be no more than the half number of 1/-1"
        assert np.sum(tensor == -1) > half_num_test / 2, "test set should be no more than the half number of 1/-1"

        # select idx to be changed as positive
        original_p_idx = np.nonzero(tensor == 1)
        p_idx_tmp = np.random.choice(range(len(original_p_idx[0])), half_num_test, replace=False)
        p_idxs = tuple(original_p_idx[k][p_idx_tmp] for k in range(3))

        # select idx to be changed as negative
        original_n_idx = np.nonzero(tensor == -1)
        n_idx_tmp = np.random.choice(range(len(original_n_idx[0])), half_num_test, replace=False)
        n_idxs = tuple(original_n_idx[k][n_idx_tmp] for k in range(3))

        # if mask[idx]==True, then tensor[idx] is in test set
        test_mask = np.zeros(shape=tensor.shape, dtype=bool)
        test_mask[p_idxs] = True
        test_mask[n_idxs] = True
    training_set = np.copy(tensor)
    training_set[test_mask] = 0
    return {'training_set': training_set, 'test_mask': test_mask}
