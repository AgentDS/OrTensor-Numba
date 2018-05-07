#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/5 下午11:34
# @Author  : Shiloh Leung
# @Site    : 
# @File    : basic.py
# @Software: PyCharm
import numpy as np
import math
from scipy.special import expit
from numba import int8, int16, int32, float32, float64
from numba import jit, prange
from OrTensor._numba.post_accumulator_numba import posterior_accumulator

"""
Each element in tensor X is in {-1, 0, 1}.
Only missing data is coded as 0.
"""


@jit('int8(int8[:], int8[:], int8[:])', nopython=True, nogil=True)
def Vector_Inner_product(A_i, B_j, C_k):
    """
    Calculate the Boolean product of three vectors of the same length.

    :param A_i: R, the i_{th} row in factor matrix A
    :param B_j: R, the j_{th} row in factor matrix B
    :param C_k: R, the k_{th} row in factor matrix C
    :return: 2*min(1, sum(A_i.*B_j.*C_k)) - 1
    """
    R = A_i.shape[0]
    # if R <=10,000,000, use range(), else use prange()
    for r in range(R):
        if A_i[r] * B_j[r] * C_k[r] == 1:
            return 1
    return -1


@jit('int8[:,:,:](int8[:], int8[:], int8[:])', nopython=False, nogil=True, parallel=True)
def Boolean_Vector_Outer_product(a, b, c):
    """
    Return the result of outer product of 3 vectors via boolean operations.

    :param a: I, boolean vector
    :param b: J, boolean vector
    :param c: K, boolean vector
    :return: IxJxK, Boolean tensor
    """
    I = a.shape[0]
    J = b.shape[0]
    K = c.shape[0]
    X = np.zeros([I, J, K], dtype=np.int8)
    # if I, J, K is not too large (less than 10,000) use at least 1 numba.prange
    for i in prange(I):
        for j in prange(J):
            for k in range(K):
                if a[i] * b[j] * c[k] == 1:
                    X[i, j, k] = 1
    return X


@jit('int8(float64, int8)', nopython=True, nogil=True)
def flip_metropolized_Gibbs_numba(prob, a):
    """
    Flip a according to metropolized Gibbs with probability prob/(1-prob)

    :param logit_prob: float64, the logit probability of a=1
    :param a: int8, single element in the factor matrix
    :return: final a (-1 or 1)
    """
    if a == 1:
        if prob <= 0.5:
            return -1
        else:
            alpha = (1 - prob) / prob
    else:
        if prob > 0.5:
            return 1
        else:
            alpha = prob / (1 - prob)
    if np.random.rand() < alpha:
        return -a
    else:
        return a

