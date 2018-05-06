#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/6 下午12:05
# @Author  : Shiloh Leung
# @Site    : 
# @File    : time_test.py
# @Software: PyCharm
from OrTensor._numba.basic import Triple_Boolean_Inner_product as inner
from OrTensor._numba.basic import Triple_Boolean_Inner_product2 as inner2
import time
import numpy as np

def test_inner_prod_time():
    R = 1000
    a = np.zeros(R, dtype=np.int8)
    b = np.zeros(R, dtype=np.int8)
    c = np.zeros(R, dtype=np.int8)
    a[R - 2] = 1
    b[R - 2] = 1
    c[R - 2] = 1
    print("Using numba.prange():  ", end='')
    start1 = time.time()
    res1 = inner(a, b, c)
    end1 = time.time()
    print("ans = %d, time = %.8fs" % (res1, end1 - start1))

    print("Using range():  ", end='')
    start2 = time.time()
    res2 = inner2(a, b, c)
    end2 = time.time()
    print("ans = %d, time = %.8fs" % (res2, end2 - start2))



if __name__=='__main__':
    test_inner_prod_time()