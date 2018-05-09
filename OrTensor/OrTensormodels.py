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
        self.sampling_fct = None
        self.val = val
        self.fixed = fixed
        self.beta_prior = (1, 1)

    def print_value(self):
        return '\t'.join([str(round(expit(np.mean(x)), 3))
                          for x in [self.val]])


class OTMatrix(OTTrace):
    def __init__(self, shape=None, val=None,
                 bernoulli_prior=0.5,
                 child_axis=None, fixed=False):
        self.trace_idx = 0
        self.sampling_fct = None
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


class OTTensor(OTTrace):
    def __init__(self, shape=None, val=None, fixed=False):
        self.trace_idx = 0
        self.parents = []
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
        self.shape = shape

    def __call__(self):
        return self.val

    def display(self, axis, method='mean'):
        if method == 'mean':
            if axis == 0:
                I = self.shape[0]
                for i in range(I):
                    lib.plot_matrix(self.val[i, :, :].mean())
            elif axis == 1:
                J = self.shape[1]
                for j in range(J):
                    lib.plot_matrix(self.val[:, j, :].mean())
            elif axis == 2:
                K = self.shape[2]
                for k in range(K):
                    lib.plot_matrix(self.val[:, :, k].mean())
            else:
                raise ValueError("'axis' can only be 0, 1 or 2")

        elif method == 'map':
            if axis == 0:
                I = self.shape[0]
                for i in range(I):
                    lib.plot_matrix(self.val[i, :, :].mean() > 0.5)
            elif axis == 1:
                J = self.shape[1]
                for j in range(J):
                    lib.plot_matrix(self.val[:, j, :].mean() > 0.5)
            elif axis == 2:
                K = self.shape[2]
                for k in range(K):
                    lib.plot_matrix(self.val[:, :, k].mean() > 0.5)
            else:
                raise ValueError("'axis' can only be 0, 1 or 2")
        elif method == 'original':
            if axis == 0:
                I = self.shape[0]
                for i in range(I):
                    lib.plot_matrix(self.val[i, :, :])
            elif axis == 1:
                J = self.shape[1]
                for j in range(J):
                    lib.plot_matrix(self.val[:, j, :])
            elif axis == 2:
                K = self.shape[2]
                for k in range(K):
                    lib.plot_matrix(self.val[:, :, k])
            else:
                raise ValueError("'axis' can only be 0, 1 or 2")

        else:
            raise ValueError("only support 'mean', 'map', 'original' methods.")

    @property
    def model(self):
        if 'layer' in self.__dict__.keys() and self.layer is not None:
            return self.layer.model
        else:
            return None

    def set_to_map(self):
        """
        Map sef.val to {-1, 1}.
        """
        self.val = np.array(self.mean() > 0, dtype=np.int8)
        self.val[self.val == 0] = -1


class OTLayer():
    def __init__(self, factors, lbda: OTParameter, child: OTTensor):
        """

        :param factors: list of OTMatrix, 3 factor matrices, IxR, JxR, KxR
        :param lbda: OTParameter
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
        self.auto_clean = False
        self.auto_reset = False
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
            - 'Factor-MAP' ????
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


class OTModel():
    """
    Package matrices, parameters and inference methods.
    """

    def __init__(self):
        """
        Initialize OrTensor model.
        """
        self.layer = None
        self.tensor = None
        self.matrices = []
        self.anneal = False

    @property
    def members(self):
        """
        Return all matrices in the layer.
        """
        return [self.tensor, self.layer.A, self.layer.B, self.layer.C]

    @property
    def lbda(self):
        return self.layer.lbda

    def add_layer(self, rank=None, child=None, shape=None):
        """

        :param rank: rank of decomposition mode;
        :param child: tensor of shape IxJxK
        :param shape: (I, J, K)
        :return:
        """
        # determine seize of all members
        if child is None and shape is not None:
            child = OTTensor(shape=shape)
        elif shape is None and child is not None:
            shape = child().shape
        else:
            raise ValueError("Need shape information or child (original tensor) information at least.")

        # initialize factor matrices
        factors = [OTMatrix(shape=(I, rank), child_axis=idx) for idx, I in enumerate(shape)]

        # initialize lambda
        lbda_init = 0.05
        lbda = OTParameter(val=lbda_init)

        # initialize layer object
        layer = OTLayer(factors, lbda, child)
        self.layer = layer
        return layer

    def add_tensor(self, val=None, shape=None):
        if val is not None and val.dtype != np.int8:
            val = np.array(val, dtype=np.int8)

        tensor = OTTensor(shape, val)
        self.tensor = tensor
        return tensor

    def burn_in(self, matrices, lbdas, tol=1e-2,
                convergence_window=15,
                burn_in_min=0,
                burn_in_max=2000,
                print_internal=10,
                fix_lbda_iters=0):
        """
        draw samples without saving to traces and check for convergence.
        There is an additional pre-burn-in phase where
        do not need to check for convergence.
        Convergence is detected by comparing means of noise parameters.

        :param matrices:
        :param lbda:
        :param tol:
        :param convergence_window:
        :param burn_in_min:
        :param burn_in_max:
        :param print_internal:
        :param fix_lbda_iters:
        :return:
        """
        # pre-burn-in phase
        pre_burn_in_iter = 0
        while True:
            if pre_burn_in_iter == burn_in_min:
                break
            pre_burn_in_iter += 1
            if pre_burn_in_iter % print_internal == 0:
                print('\r\titeration: ' +
                      str(pre_burn_in_iter) +
                      ' disperion.: ' +
                      '\t--\t '.join([x.print_value() for x in lbdas]),
                      end='')
            # draw samples
            [mat.sampling_fct(mat) for mat in np.random.permutation(matrices)]
            # update lambda
            if pre_burn_in_iter > fix_lbda_iters:
                if self.anneal is False:
                    [lbda.sampling_fct(lbda) for lbda in lbdas]

                # Anneal lambda for pre_burn_in_iter steps to
                # it's initially given value.
                elif self.anneal is True:
                    try:
                        assert fix_lbda_iters == 0
                    except:
                        raise ValueError('fix_lbda_iters should be zero for annealing.')
                    # pre-compute annealing steps
                    if pre_burn_in_iter == fix_lbda_iters + 1:
                        annealing_lbdas = [np.arange(
                            lbda() / burn_in_min,
                            lbda() + 2 * lbda() / burn_in_min,
                            lbda() / burn_in_min)
                            for lbda in lbdas]

                    for lbda_idx, lbda in enumerate(lbdas):
                        lbda.val = annealing_lbdas[lbda_idx][pre_burn_in_iter]

        # allocate array for lambda traces for burn in detection
        for lbda in lbdas:
            lbda.allocate_trace_arrays(convergence_window)
            lbda.trace_index = 0  # reset trace index

        # now cont. burn in and check for convergence
        burn_in_iter = 0
        while True:
            burn_in_iter += 1

            # print diagnostics
            if burn_in_iter % print_internal == 0:
                print('\r\titeration: ' +
                      str(pre_burn_in_iter + burn_in_iter) +
                      ' recon acc.: ' +
                      '\t--\t '.join([x.print_value() for x in lbdas]),
                      end='')

            #  check convergence every convergence_window iterations
            if burn_in_iter % convergence_window == 0:
                # reset trace index
                for lbda in lbdas:
                    lbda.trace_index = 0

                # check convergence for all lbdas
                if np.all([x.check_convergence(tol=tol) for x in lbdas]):
                    print('\n\tconverged at reconstr. accuracy: ' +
                          '\t--\t'.join([x.print_value() for x in lbdas]))

                    # TODO: make this nice and pretty.
                    # check for dimensions to be removed and restart burn-in if
                    # layer.auto_clean_up is True
                    if self.layer.auto_clean is True or self.layer.auto_reset is True:
                        if np.any([lib.clean_up_codes(self.layer,
                                                      self.layer.auto_reset,
                                                      self.layer.auto_clean)]):

                            for lbda in lbdas:
                                # reallocate arrays for lbda trace
                                lbda.allocate_trace_arrays(convergence_window)
                                lbda.trace_index = 0

                        else:
                            # save nu of burn in iters
                            self.burn_in_iters = burn_in_iter + pre_burn_in_iter
                            break

            # stop if max number of burn in inters is reached
            if (burn_in_iter + pre_burn_in_iter) > burn_in_max:

                # clean up non-converged auto-reset dimensions
                if self.layer.auto_reset is True:
                    lib.clean_up_codes(self.layer, reset=False, clean=True)

                print('\n\tmax burn-in iterations reached without convergence')
                # reset trace index
                for lbda in lbdas:
                    lbda.trace_index = 0
                self.burn_in_iters = burn_in_iter
                break

            # draw samples # shuffle(mats)
            [mat.sampling_fct(mat) for mat in np.random.permutation(matrices)]
            [lbda.sampling_fct(lbda) for lbda in lbdas]
            [x.update_trace() for x in lbdas]
