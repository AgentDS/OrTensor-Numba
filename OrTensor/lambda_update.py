#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/9 下午9:36
# @Author  : Shiloh Leung
# @Site    : 
# @File    : lambda_update.py
# @Software: PyCharm
import numpy as np
import OrTensor.basic_numba as basic

def get_update_fct(parm):
    print("Fitching update function: " + parm.layer.__repr__())
    if parm.sampling_fct is not None:
        return parm.sampling_fct
    return basic.lambda_update
