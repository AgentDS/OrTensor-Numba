#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/5 下午3:29
# @Author  : Shiloh Leung
# @Site    : 
# @File    : auxiliary_lib.py
# @Software: PyCharm
from OrTensor._numba.lambda_update_numba import Matrix_product
from OrTensor._numba.basic import Vector_Inner_product
import numpy as np
from scipy.special import expit
from scipy.special import logit
from scipy.special import logsumexp





def generate_data(A, B, C):
    """

    :param A: IxR
    :param B: JxR
    :param C: KxR
    :return: Tensor X, IxJxK, mapped in [-1, 1]
    """
    return Matrix_product(A, B, C)


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


def check_converge_trace(trace, tol):
    """
    compare the mean of first and second half of a sequence,
    check whether there difference is > tolerance.
    """
    r = int(len(trace) / 2)
