#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/5 下午10:27
# @Author  : Shiloh Leung
# @Site    : 
# @File    : lambda_update_numba.py
# @Software: PyCharm
import numpy as np
import math
from numba import int8, int16, int32, float32, float64
from numba import jit, prange
from OrTensor._numba.basic import Vector_Inner_product


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
                if X[i, j, k] == Boolean_Vector_Inner_product(A[i, :], B[j, :], C[k, :]):
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
