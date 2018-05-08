#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/5 下午11:34
# @Author  : Shiloh Leung
# @Site    : 
# @File    : basic_numba.py
# @Software: PyCharm
import numpy as np
from scipy.special import expit
from numba import int8, int16, int32, float32, float64
from numba import jit, prange

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


@jit('float64(float64[:], float64[:], float64[:])', nogil=True, nopython=True)
def Vector_Inner_product_fuzzy(A_i, B_j, C_k):
    """
    Compute probability of emitting a zero for fuzzy vectors under OR-AND logic.
    """
    out = np.float64(1.0)
    R = A_i.shape[0]
    for r in range(R):
        out *= 1 - (A_i[r] * B_j[r] * C_k[r])
    return 1 - out


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


@jit('int8[:,:,:](int8[:,:], int8[:,:], int8[:,:])', nogil=False, nopython=False, parallel=True)
def Matrix_product(A, B, C):
    """

    :param A: IxR
    :param B: JxR
    :param C: KxR
    :return: Tensor X, IxJxK, mapped in [-1, 1]
    """
    I = A.shape[0]
    J = B.shape[0]
    K = C.shape[0]
    X = np.zeros([I, J, K], dtype=np.int8)
    for i in prange(I):
        for j in prange(J):
            for k in range(K):
                X[i, j, k] = Vector_Inner_product(A[i, :], B[j, :], C[k, :])
    return X


@jit('float64[:,:](float64[:,:], float64[:,:], float64[:,:])', nogil=True, nopython=False, parallel=True)
def Matrix_product_fuzzy(A, B, C):
    I = A.shape[0]
    J = B.shape[0]
    K = C.shape[0]
    X = np.zeros([I, J, K], dtype=np.float64)
    for i in prange(I):
        for j in prange(J):
            for k in range(M):
                X[i, j, k] = Vector_Inner_product_fuzzy(A[i, :], B[j, :], C[k, :])
    return X


# post_accumulator_numba
@jit('int16(int8[:,:], int8[:,:], int8[:], int8[:,:], int16)', nopython=True, nogil=True)
def posterior_accumulator(B, C, A_i, X_i, r):
    """
    Calculate the accumulator in Algorithm 1 (computation
    of the full conditional of a_{ir}). Refer to my paper
    for more details.

    :param B: JxR, factor matrix B
    :param C: KxR, factor matrix C
    :param A_i: R, the i_{th} row of factor matrix A
    :param X_i: JxK
    :param r: index of feature
    :return:
    """
    J, R = B.shape
    K, _ = C.shape

    accumulator = 0
    for j in range(J):
        for k in range(K):
            if B[j, r] * C[k, r] == 0:
                continue

            flag = False
            for r_ in range(R):
                if (r_ != r) and (A_i[r_] * B[j, r_] * C[k, r_] == 1):
                    flag = True
                    break
            if flag is False:
                accumulator += X_i[j, k]
    return accumulator


@jit('int64(int8[:,:], int8[:,:], int8[:,:], int8[:,:,:])', nogil=True, nopython=True, parallel=True)
def correct_prediction(A, B, C, X):
    """

    :param A: IxR, factor matrix A
    :param B: JxR, factor matrix B
    :param C: KxR, factor matrix C
    :param X: IxJxK, original tensor
    :return: the number of elements predicted by model correctly
    """
    I, J, K = X.shape
    P = np.int64(0)
    for i in prange(I):
        for j in prange(J):
            for k in range(K):
                if X[i, j, k] == Vector_Inner_product(A[i, :], B[j, :], C[k, :]):
                    P += 1
    return P


def lambda_update(parm):
    """
    Set lambda to its MLE:
    $\\frac{1}{1+e^{-\\lambda}}=\\frac{P}{IJK}$

    :param parm:
    :return:
    """

    P = correct_prediction(*[factor.val for factor in parm.layer.factors], parm.layer.child())
    IJK = np.prod(parm.layer.child().shape) - np.count_nonzero(parm.layer.child() == 0)
    parm.val = np.max([0, np.min([1000, -np.log(IJK / float(P) - 1)])])
