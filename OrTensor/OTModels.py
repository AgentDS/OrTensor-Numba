#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/6 下午11:12
# @Author  : Shiloh Leung
# @Site    : 
# @File    : OTModels.py
# @Software: PyCharm

"""
Models design for OrTensor, including
"""
import numpy as np
import OrTensor.auxiliary_lib as lib
import OrTensor.basic_numba as basic
from numpy.random import rand


class OTTrace():
    """
    posterior traces arrays
    Inherited to OTMatrix and OTParameter.
    """

    def __call__(self):
        return self.val

    def allocate_trace_arrays(self, num_samples):
        num_samples = int(num_samples)
        self.trace = np.empty([num_samples] + [x for x in self.val.shape], dtype=np.int8)

    def update_trace(self):
        self.trace[self.trace_idx] = self.val
        self.trace_idx += 1

    def mean(self):
        return np.mean(self.trace, axis=0)

    def check_convergence(self, tol):
        if self.trace.ndim == 1:
            return lib.check_converge_single_trace(self.trace, tol)
        elif self.trace.ndim == 2:
            check_res = np.all(
                [lib.check_converge_single_trace(self.trace[:, r], tol) for r in range(self.trace.shape[1])])
            return check_res


class OTParameter(OTTrace):
    """
    Basic class for parameters (lambda)
    """

    def __init__(self, val, fixed=False):
        self.trace_idx = 0
        self.sampling_func = None
        self.val = val
        self.fixed = fixed
        self.beta_prior = (1, 1)


class OTMatrix(OTTrace):
    def __init__(self, shape=None, val=None,
                 bernoulli_prior=0.5,
                 child_axis=None, fixed=False):
        self.trace_idx = 0
        self.sampling_func = None
        self.child_axis = child_axis  # index of factor matrix (0,1 or 2)
        self.parents = []
        self.bernoulli_prior = bernoulli_prior
        if val is not None:
            shape = val.shape
        # Elements of self.val are all mapped in {-1, 1}
        # if already given value, assign it
        if type(val) is np.ndarray:
            self.val = np.array(val, dtype=np.int8)
        elif type(val) is float:
            self.val = 2 * np.array(rand(*shape) > val, dtype=np.int8) - 1
        # otherwise generate bernoulli random data
        else:
            self.val = 2 * np.array(rand(*shape) > 0.5, dtype=np.int8) - 1

        self.fixed = fixed
        # fix some matrix entries
        self.fixed_entries = np.zeros(self().shape, dtype=np.int8)

        self.layer = None

    def __call__(self):
        return self.val

    def display(self, method='mean'):
        if method == 'mean':
            lib.plot_matrix(self.mean())
        elif method == 'map':
            lib.plot_matrix(self.mean() > 0.5)
        elif method == 'original':
            lib.plot_matrix(self())
        else:
            raise ValueError("only support 'mean', 'map', 'original' methods.")

    @property
    def model(self):
        if 'layer' in self.__dict__.keys() and self.layer is not None:
            return self.layer.model
        else:
            return None

    @property
    def siblings(self):
        """
        Retuen other 2 factor matrices in child_axis order.

        :return: [np.ndarray, np.ndarray], 2 siblings of self factor matrix
        """
        siblings = [mat for mat in self.layer.factors if mat is not self]
        return sorted(siblings, key=lambda mat: mat.child_axis)

    def set_to_map(self):
        """
        Map sef.val to {-1, 1}.
        """
        self.val = np.array(self.mean() > 0, dtype=np.int8)
        self.val[self.val == 0] = -1



