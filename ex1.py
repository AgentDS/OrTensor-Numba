#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/11 下午10:10
# @Author  : Shiloh Leung
# @Site    : 
# @File    : ex1.py
# @Software: PyCharm
import numpy as np
import experiment_tool as extool
import OrTensor.auxiliary_lib as lib
import matplotlib.pyplot as plt
from OrTensor.OrTensormodels import OTModel
from OrTensor.OrTensormodels import OTTensor
from OrTensor.OrTensormodels import OTLayer
from OrTensor.OrTensormodels import OTParameter
from OrTensor.basic_numba import posterior_accumulator as post_acc

# np.random.seed(3)
I = 10
J = 4
K = 5
R = 3
A = np.array(np.random.rand(I, R) > .5, dtype=np.int8)
B = np.array(np.random.rand(J, R) > .5, dtype=np.int8)
C = np.array(np.random.rand(K, R) > .5, dtype=np.int8)
X = lib.generate_data(2 * A - 1, 2 * B - 1, 2 * C - 1)  # take Boolean product
X_noisy = lib.add_bernoulli_noise(X, p=.1)
otm = OTModel()
data = otm.add_tensor(X_noisy)
layer = otm.add_layer(rank=R, child=data)
# # Fix particular entries (1s in fixed_entries matrix) (optional)
# layer.factors[1].fixed_entries = np.zeros(layer.factors[1]().shape, dtype=np.int8)
# layer.factors[1].fixed_entries[0,:] = 1
#
# # Set priors beta prior on sigmoid(lambda) (optional)
# layer.lbda.beta_prior = (1,1)
#
# # Set iid bernoulli priors on factor matrix entries (optional)
# layer.factors[1].bernoulli_prior = .5


# run inference
otm.infer(num_samples=50, burn_in_min=100, burn_in_max=1000)
# inspect the factor mean
[layer.factors[i].show() for i in range(len(layer.factors))]
# inspect the reconstruction
fig, ax = lib.plot_matrix(X_noisy)
ax.set_title('Input data')

fig, ax = lib.plot_matrix(layer.output(method='factor_map'))
ax.set_title('Reconstruction')

fig, ax = lib.plot_matrix(X)
ax.set_title('Noisefree data')
plt.show()