#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/5 下午3:29
# @Author  : Shiloh Leung
# @Site    : 
# @File    : auxiliary_lib.py
# @Software: PyCharm
from OrTensor.basic_numba import Matrix_product
from OrTensor.basic_numba import Matrix_product_fuzzy
from OrTensor.basic_numba import Vector_Inner_product
import numpy as np
import seaborn
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import itertools
from scipy.special import expit
from scipy.special import logit
from scipy.special import logsumexp


def plot_matrix_ax(matrix, ax):
    if np.any(matrix < 0):
        print('rescaling matrix to probabilities...')
        matrix = 0.5 * (matrix + 1)
    color_map = seaborn.cubehelix_palette(8, start=2, light=0.5, dark=0, reverse=False, as_cmap=True)
    seaborn.set_style("whitegrid", {'axes,grid': False})
    cax = ax.imshow(matrix, aspext='auto', cmap=color_map, vmin=0, vmax=1)
    return ax, cax


def plot_matrix(matrix, fig_size=(7, 4), draw_cbar=True, vmin=0, vmax=1, cmap=None):
    if np.any(matrix < 0):
        print('rescaling matrix to probabilities')
        matrix = 0.5 * (matrix + 1)
    if cmap is None:
        cmap = seaborn.cubehelix_palette(8, start=2, dark=0, light=1, reverse=False, as_cmap=True)

    seaborn.set_style("whitegrid", {'axes.grid': False})

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    cax = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')

    if draw_cbar is True:
        fig.colorbar(cax, orientation='vertical')

    return fig, ax


def plot_codes(matrix):
    """

    :param factor: factor matrix
    :return:
    """
    color_map = seaborn.cubehelix_palette(8, start=2, dark=0, light=1, reverse=False, as_cmap=True)
    seaborn.set_style("whitegrid", {'axes.grid': False})

    r_idx = np.argsort(-matrix.layer.lbda()[:-1])

    fig = plt.figure(figsize=(7, 4))
    ax_codes = fig.add_subplot(111)
    ax_codes.imshow(matrix.mean().transpose()[r_idx, :], aspect='auto', cmap=color_map)
    ax_codes.set_yticks(range(matrix().shape[1]))

    return fig, ax_codes


def compute_roc_auc(data, data_train, prediction):
    """
    Compute the area under ROC curve

    :param data:
    :param data_train:
    :param prediction:
    :return:
    """
    zero_idx = np.where(data_train == 0)
    zero_idx = zip(list(zero_idx)[0], list(zero_idx)[1], list(zero_idx)[2])
    auc = roc_auc_score([data[i, j, k] for i, j, k in zero_idx], [prediction[i, j, k] for i, j, k in zero_idx])
    return auc


def split_test_train(data, p=0.1):
    """
    In a binary matrix {-1,1}, set randomly
    p/2 of the 1s and p/2 of the -1s to 0.
    To create a test set.

    :param data:
    :param p:
    :return:
    """
    if -1 not in np.unique(data):
        data = 2 * data - 1
    num_zeros = np.prod(data.shape) * p
    idx_pairs = list(itertools.product(range(data.shape[0]), range(data.shape[1]), range(data.shape[2])))

    # randomly set same number -1/1 data as unobserved (coded as 0)
    true_idx_pairs = [idx for idx in idx_pairs if data[idx] == 1]
    false_idx_pairs = [idx for idx in idx_pairs if data[idx] == -1]
    true_num = len(true_idx_pairs)
    false_num = len(false_idx_pairs)
    true_random_idx = np.random.choice(range(true_num), int(num_zeros / 2), replace=False)
    false_random_idx = np.random.choice(range(false_num), int(num_zeros / 2), replace=False)
    data_train = data.copy()
    for i, j, k in true_random_idx:
        data_train[i, j, k] = 0
    for i, j, k in false_random_idx:
        data_train[i, j, k] = 0

    return data_train


def generate_data(A, B, C, fuzzy=False):
    """

    :param A: IxR
    :param B: JxR
    :param C: KxR
    :return: Tensor X, IxJxK, mapped in [-1, 1]
    """
    if fuzzy is False:
        return Matrix_product(A, B, C)
    else:
        return Matrix_product_fuzzy(np.array(A, dtype=np.float64),
                                    np.array(B, dtype=np.float64),
                                    np.array(C, dtype=np.float64))


def add_bernoulli_noise(X, p):
    """

    :param X: IxJxK, original tensor
    :param p: float, probability of Bernoulli distribution
    :return: IxJxK, tensor with noise
    """
    X_ = X.copy()
    I, J, K = X.shape
    for i in range(I):
        for j in range(J):
            for k in range(K):
                if rand() < p:
                    X_[i, j, k] = -X[i, j, k]
    return X_


def Boolean_Matrix_product(A, B, C):
    """

    :param A: IxR
    :param B: JxR
    :param C: KxR
    :return: IxJxK, Boolean tensor X from A, B, C
    """
    I = A.shape[0]
    J = B.shape[0]
    K = C.shape[0]
    R = A.shape[1]
    X = np.zeros(shape=(I, J, K), dtype=bool)
    assert (B.shape[1] == R)
    assert (C.shape[1] == R)

    for i in range(I):
        for j in range(J):
            for k in range(K):
                if Vector_Inner_product(A[i, :], B[j, :], C[k, :]) == 1:
                    X[i, j, k] = True
    return X


def check_bin_coding(data):
    """
    Code data in {-1,1}. Check and correct the coding here.

    :param data:
    :return: np.int8, corrected data mapped in {-1,1}
    """
    if -1 not in np.unique(data):
        data = 2 * data - 1
    return np.array(data, dtype=np.int8)


def check_converge_single_trace(trace, tol):
    """
    compare the mean of first and second half of a sequence,
    check whether there difference is > tolerance.
    """
    r = int(len(trace) / 2)
    first = expit(np.mean(trace[:r]))
    second = expit(np.mean(trace[r:]))
    whole = expit(np.mean(trace))
    if np.abs(first - second) < tol:
        return True
    else:
        return False


def clean_up_codes(layer, reset=True, clean=False):
    """
    Remove redundant or all-zero latent dimensions
    from layer and adjust all attributes accordingly.
    Return True, if any dimension was removed, False otherwise.
    """

    if reset is True:
        cleaning_action = reset_dimension
    elif clean is True:
        cleaning_action = remove_dimension

    reduction_applied = False
    # remove inactive codes
    r = 0
    while r < layer.size:  # need to use while loop because layer.size changes.
        if np.any([np.all(f()[:, r] == -1) for f in layer.factors]):
            cleaning_action(r, layer)
            reduction_applied = True
        r += 1

    # remove duplicates
    r = 0
    while r < layer.size:
        r_ = r + 1
        while r_ < layer.size:
            for f in layer.factors:
                if np.all(f()[:, r] == f()[:, r_]):
                    reduction_applied = True
                    cleaning_action(r_, layer)
                    break
            r_ += 1
        r += 1

    if reduction_applied is True:
        if reset is True:
            print('\n\tre-initialise duplicate or useless latent ' +
                  'dimensions and restart burn-in. New L=' + str(layer.size))

        elif clean is True:
            print('\n\tremove duplicate or useless latent ' +
                  'dimensions and restart burn-in. New L=' + str(layer.size))

    return reduction_applied


def remove_dimension(r_, layer):
    # update for tensorm link does not support parents
    # nor priors
    # layer.size -= 1
    for f in layer.factors:
        f.val = np.delete(f.val, r_, axis=1)


def reset_dimension(r_, layer):
    for f in layer.factors:
        f.val[:, r_] = -1
