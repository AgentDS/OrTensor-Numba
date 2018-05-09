#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/6 下午11:12
# @Author  : Shiloh Leung
# @Site    : 
# @File    : OrTensormodels.py
# @Software: PyCharm

"""
Models design for OrTensor, including
"""
import numpy as np
from numpy.random import rand
from scipy.special import expit
from scipy.special import logit
import OrTensor.auxiliary_lib as lib
import OrTensor.basic_numba as basic


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
        Return other 2 factor matrices in child_axis order.

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


class OTLayer():
    def __init__(self, factors, lbda, child):
        """

        :param factors: 3 factor matrices, IxR, JxR, KxR
        :param lbda: vector, containing R elements
        :param child: IxJxK, tensor which can be constructed from 3 factor matrices
        """
        self.factors = sorted(factors, key=lambda mat: mat.child_axis)
        self.lbda = lbda
        self.lbda.layer = self
        self.child = child
        # TODO: append(self) or append(factors)??
        self.child.parents.append(self)
        for factor in factors:
            factor.layer = self
        self.prediction = None

    @property
    def A(self):
        return self.factors[0]

    @property
    def B(self):
        return self.factors[1]

    @property
    def C(self):
        return self.factors[2]

    @property
    def __iter__(self):
        return iter(self.factors)

    @property
    def __repr__(self):
        parents_string = ', '.join([str(mat.shape[0]) + 'x' + str(mat.shape[1]) for mat in self.factors])
        child_string = 'x'.join([str(mat.shape[0]) for mat in self.factors])
        return '<OrTensor: ' + parents_string + ' -> ' + child_string + '>'

    def output(self,
               method='factor_map',
               noisy=False,
               lazy=False,
               map_to_prob=True):
        """
        Compute output matrix from posterior samples.
        Valid methods are:
            - 'point_estimate'
                output of the current state of factors
            - 'Factor-MAP' TODO
                From the posterior MAP of factors
            - 'Factor-MEAN'
                Computed from posterior mean of factors

        Note, that outputs are always probabilities in (0,1)

        :param method:
        :param noisy:
        :param map_to_prob:
        :return:
        """
        if type(self.prediction) is np.ndarray and lazy is True:
            print('returning previously computed value ' +
                  'under disregard of technique.')
            return self.prediction
        if method == 'poit_estimate':
            out_tensor = lib.generate_data(self.A, self.B, self.C)
            out_tensor = (1 + out_tensor) * 0.5
        elif method == 'factor_map':
            tmpA = 2 * (self.A.mean() > 0) - 1
            tmpB = 2 * (self.B.mean() > 0) - 1
            tmpC = 2 * (self.C.mean() > 0) - 1
            out_tensor = lib.generate_data(tmpA, tmpB, tmpC)
            out_tensor = np.array(out_tensor == 1, dtype=np.int8)
        elif method == 'factor_mean':
            # output does not need to be mapped to probabilities
            tmpA = 0.5 * (self.A.mean() + 1)
            tmpB = 0.5 * (self.B.mean() + 1)
            tmpC = 0.5 * (self.C.mean() + 1)
            out_tensor = lib.generate_data(tmpA, tmpB, tmpC,  # map to (0,1)
                                           fuzzy=True)

        # convert to probability of generating 1
        if noisy is True:
            out_tensor = out_tensor * expit(self.lbda.mean()) + \
                         (1 - out_tensor) * lib.expit(-self.lbda.mean())
        self.prediction = out_tensor

        if map_to_prob is True:
            return out_tensor
        else:
            return 2 * out_tensor - 1

