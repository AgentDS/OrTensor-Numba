#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/5 下午10:29
# @Author  : Shiloh Leung
# @Site    : 
# @File    : post_accumulator_numba.py
# @Software: PyCharm
import numpy as np
from numba import jit


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
                if (r_ != r) and (A_i[r_]*B[j,r_]*C[k,r_]==1):
                    flag = True
                    break
            if flag is False:
                accumulator += X_i[j,k]
    return accumulator

