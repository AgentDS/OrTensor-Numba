#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/10 下午10:19
# @Author  : Shiloh Leung
# @Site    : 
# @File    : experiment_tool.py
# @Software: PyCharm
import numpy as np
def index_generator(tensor):
    return tuple([np.random.randint(dim) for dim in tensor.shape])

def split_train_test()