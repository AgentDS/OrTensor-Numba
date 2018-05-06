#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/5 下午3:29
# @Author  : Shiloh Leung
# @Site    : 
# @File    : auxiliary_lib.py
# @Software: PyCharm
from OrTensor._numba.lambda_update_numba import Boolean_Matrix_product

def generate_data(A, B, C):
    """

    :param A: IxR
    :param B: JxR
    :param C: KxR
    :return: Tensor X, IxJxK, mapped in [-1, 1]
    """
    return Boolean_Matrix_product(A, B, C)




