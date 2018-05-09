#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/8 上午11:13
# @Author  : Shiloh Leung
# @Site    : 
# @File    : matrix_update.py
# @Software: PyCharm
import numpy as np
import OrTensor.basic_numba as basic
from scipy.special import logit


def get_sampling_fct(mat):
    if mat.sampling_fct is not None:
        return mat.sampling_fct
    transpose_order = tuple([mat.child_axis] + [sib.child_axis for sib in mat.siblings])

    logit_bernoulli_prior = np.float64(logit(mat.bernoulli_prior))

    # standard case: one child, no parents
    def LOM_sampler(mat):
        basic.sampling_fct(
            mat(),
            mat.fixed_entries,
            *[x() for x in mat.siblings],
            mat.layer.child().transpose(transpose_order),
            mat.layer.lbda(),
            logit_bernoulli_prior)

    return LOM_sampler
