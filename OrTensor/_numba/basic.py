#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/5 下午11:34
# @Author  : Shiloh Leung
# @Site    : 
# @File    : basic.py
# @Software: PyCharm
import numpy as np
import numba
from numba import jit, prange

"""
Each element in tensor X is in {-1, 0, 1}.
Only missing data is coded as 0.
"""

@jit('int8(int8[:], int8[:], int8[:])', nopython=True, nogil=True)
def Triple_Boolean_Inner_product(A_i, B_j, C_k):
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
        if A_i[r]*B_j[r]*C_k[r] == 1:
            return 1
    return -1


# @jit('int8(int8[:], int8[:], int8[:])', nopython=True, nogil=True)
# def Triple_Boolean_Inner_product2(A_i, B_j, C_k):
#     """
#     Calculate the Boolean product of three vectors of the same length.
#
#     :param A_i: R, the i_{th} row in factor matrix A
#     :param B_j: R, the j_{th} row in factor matrix B
#     :param C_k: R, the k_{th} row in factor matrix C
#     :return: 2*min(1, sum(A_i.*B_j.*C_k)) - 1
#     """
#     R = A_i.shape[0]
#     for r in range(R):
#         if A_i[r]*B_j[r]*C_k[r] == 1:
#             return 1
#     return -1


# @jit()
# def Triple