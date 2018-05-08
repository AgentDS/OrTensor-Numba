#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/6 下午12:05
# @Author  : Shiloh Leung
# @Site    : 
# @File    : time_test.py
# @Software: PyCharm
# from OrTensor.basic_numba import Triple_Boolean_Inner_product as inner
# from OrTensor.basic_numba import Boolean_Outer_product as outer
import time
import numpy as np


def test_inner_prod_time():
    R = 10000000
    a = np.zeros(R, dtype=np.int8)
    b = np.zeros(R, dtype=np.int8)
    c = np.zeros(R, dtype=np.int8)
    a[R - 2] = 1
    b[R - 2] = 1
    c[R - 2] = 1
    print("Using numba.prange():  ", end='')
    start1 = time.time()
    for i in range(1000):
        res1 = inner(a, b, c)
    end1 = time.time()
    print("ans = %d, time = %.8fs" % (res1, end1 - start1))


def test_outer_prod_time():
    I, J, K = 9000, 1000, 1000
    a = np.ones(I, dtype=np.int8)
    b = np.ones(J, dtype=np.int8)
    c = np.ones(K, dtype=np.int8)

    print("Using 2 numba.prange():  ", end='')
    start1 = time.time()
    outer(a, b, c)
    end1 = time.time()
    print("time = %.8fs" % (end1 - start1))


if __name__ == '__main__':
    # test_inner_prod_time()
    # test_outer_prod_time()
    pass
